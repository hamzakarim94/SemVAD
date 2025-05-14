import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import pickle
from model import CLIPVAD
from utils.dataset import XDDataset
from utils.tools import get_batch_mask, get_prompt_text
from utils.xd_detectionMAP import getDetectionMAP as dmAP
import xd_option
def process_array(array):
    array = array.squeeze(axis=1)
    b, n = array.shape  # Get original shape

    if b < 256:
        # Pad with zeros to shape (256, 768)
        pad_size = 256 - b
        pad_matrix = np.zeros((pad_size, n))
        array = np.vstack([array, pad_matrix])
        return array.reshape(1,array.shape[0],array.shape[1]) # Shape (1, 256, 768)

    else:
        # Split into chunks of 256
        num_full_chunks = b // 256  # Number of full 256 rows
        remainder = b % 256  # Remaining rows

        # Create a list of chunks
        chunks = [array[i * 256: (i + 1) * 256] for i in range(num_full_chunks)]

        # Handle the last chunk (if remainder exists)
        if remainder > 0:
            last_chunk = np.zeros((256, n))  # Create a zero-padded chunk
            last_chunk[:remainder, :] = array[num_full_chunks * 256:]  # Fill with remaining rows
            chunks.append(last_chunk)

        return np.stack(chunks)  # Shape (num_chunks, 256, 768)
def test(model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels, device):
    
    model.to(device)
    model.eval()
    with open("xd_clip_labels.pkl", "rb") as f:
        loaded_data = pickle.load(f)
    element_logits2_stack = []

    with torch.no_grad():
        for i, item in enumerate(testdataloader):
            visual = item[0].squeeze(0)
            length = item[2]
            v_label = item[3][0]
            length = int(length)
            len_cur = length
            if len_cur < maxlen:
                visual = visual.unsqueeze(0)

            visual = visual.float().to(device)

            lengths = torch.zeros(int(length / maxlen) + 1)
            for j in range(int(length / maxlen) + 1):
                if j == 0 and length < maxlen:
                    lengths[j] = length
                elif j == 0 and length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                elif length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                else:
                    lengths[j] = length
            lengths = lengths.to(int)
            padding_mask = get_batch_mask(lengths, maxlen).to(device)
            T_feat = process_array(np.load("clip_256_xd/" + v_label + ".npy", allow_pickle=True))
            T_feat = torch.tensor(T_feat.astype(np.float32)).to(device)
            label_clip_feats = []
            for prompt in prompt_text:
                clip = loaded_data[prompt]
                label_clip_feats.append(clip)
            label_clip_feats = np.stack(label_clip_feats)
            label_clip_feats = torch.tensor(label_clip_feats).to(device)
            _,_,_,_,logits1, logits2 = model(visual, padding_mask, prompt_text, lengths,T_feat,label_clip_feats)
            logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
            logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])
            prob2 = (1 - logits2[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1))
            prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1))

            if i == 0:
                ap1 = prob1
                ap2 = prob2
            else:
                ap1 = torch.cat([ap1, prob1], dim=0)
                ap2 = torch.cat([ap2, prob2], dim=0)

            element_logits2 = logits2[0:len_cur].softmax(dim=-1).detach().cpu().numpy()
            element_logits2 = np.repeat(element_logits2, 16, 0)
            element_logits2_stack.append(element_logits2)

    ap1 = ap1.cpu().numpy()
    ap2 = ap2.cpu().numpy()
    ap1 = ap1.tolist()
    ap2 = ap2.tolist()

    ROC1 = roc_auc_score(gt, np.repeat(ap1, 16))
    AP1 = average_precision_score(gt, np.repeat(ap1, 16))
    ROC2 = roc_auc_score(gt, np.repeat(ap2, 16))
    AP2 = average_precision_score(gt, np.repeat(ap2, 16))

    print("AUC1: ", ROC1, " AP1: ", AP1)
    print("AUC2: ", ROC2, " AP2:", AP2)

    dmap, iou = dmAP(element_logits2_stack, gtsegments, gtlabels, excludeNormal=False)
    averageMAP = 0
    for i in range(5):
        print('mAP@{0:.1f} ={1:.2f}%'.format(iou[i], dmap[i]))
        averageMAP += dmap[i]
    averageMAP = averageMAP/(i+1)
    print('average MAP: {:.2f}'.format(averageMAP))

    return ROC1, AP1,ROC2,AP2 ,averageMAP,dmap#, averageMAP


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = xd_option.parser.parse_args()

    label_map = dict({'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'})

    test_dataset = XDDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device)
    model_param = torch.load(args.model_path)
    model.load_state_dict(model_param)

    test(model, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)