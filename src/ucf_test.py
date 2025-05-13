import torch
from torch import nn
import pickle
import cv2
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import auc, roc_curve, confusion_matrix, precision_recall_curve
from model import SemVAD
from utils.dataset import UCFDataset
from utils.tools import get_batch_mask, get_prompt_text
from utils.ucf_detectionMAP import getDetectionMAP as dmAP
import ucf_option
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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
def first_n_words(text, n):
    return " ".join(text.split()[:n])
def process_patches(patches):
    #patches = patches.reshape(4, 4)
    extended_grid = np.pad(patches, pad_width=1, mode='edge')

    # Step 2: Perform average pooling
    output = np.zeros_like(patches)  # Initialize output grid
    for i in range(7):  # Iterate over each cell in the 7x7 grid
        for j in range(7):
            # Define the 2x2 pooling window
            window = extended_grid[i:i + 3, j:j + 3]
            # Compute the average value
            output[i, j] = np.mean(window)
    #kernel = np.array([[1, 1], [1, 1]])/4
    # Apply convolution
    #con_patches = convolve(patches, kernel, mode='constant',cval=0.0)
    con_patches = output
    min_val = np.min(con_patches)
    max_val = np.max(con_patches)

    # Normalize the matrix to be between 0 and 1
    patches = (con_patches - min_val) / (max_val - min_val)#'''
    return patches

def calculate_intersection_and_union(x_min, x_max, y_min, y_max,
                                     x_min_patch, x_max_patch, y_min_patch, y_max_patch):
    # Intersection coordinates
    inter_x_min = max(x_min, x_min_patch)
    inter_x_max = min(x_max, x_max_patch)
    inter_y_min = max(y_min, y_min_patch)
    inter_y_max = min(y_max, y_max_patch)

    # Intersection area
    if inter_x_max > inter_x_min and inter_y_max > inter_y_min:
        intersection_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    else:
        intersection_area = 0


    area_box_patch = (x_max_patch - x_min_patch) * (y_max_patch - y_min_patch)

    # Union area
    area_outside_bouds = area_box_patch - intersection_area

    return intersection_area, area_outside_bouds



def fixed_smooth(logits, t_size):
    ins_preds = torch.zeros(0).cuda()
    assert t_size > 1
    if len(logits) % t_size != 0:
        delta = t_size - len(logits) % t_size
        logits = F.pad(logits, (0,  delta), 'constant', 0)

    seq_len = len(logits) // t_size
    for i in range(seq_len):
        seq = logits[i * t_size: (i + 1) * t_size]
        avg = torch.mean(seq, dim=0)
        avg = avg.repeat(t_size)
        ins_preds = torch.cat((ins_preds, avg))

    return ins_preds
def test(model,testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels, device):
    with open("ucf_clip_labels.pkl", "rb") as f:
        loaded_data = pickle.load(f)
    model.to(device)
    model.eval()
    element_logits2_stack = []


    with torch.no_grad():
        videos = []
        ap3= []
        for i, item in enumerate(testdataloader):
            visual = item[0].squeeze(0)
            length = item[2]
            v_label = item[3][0]
            videos.append(v_label)
            length = int(length)
            len_cur = length



            if len_cur < maxlen:
                visual = visual.unsqueeze(0)

            visual = visual.to(device)

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
            T_feat = process_array(np.load("clip_256_ucf/"+v_label+".npy",allow_pickle=True))
            T_feat = torch.tensor(T_feat.astype(np.float32)).to(device)
            label_clip_feats = []
            for prompt in prompt_text:
                clip = loaded_data[prompt]
                label_clip_feats.append(clip)
            label_clip_feats = np.stack(label_clip_feats)
            label_clip_feats = torch.tensor(label_clip_feats).to(device)
            _,_,_,_,logits1, logits2 = model(visual.float(), padding_mask, prompt_text, lengths,T_feat,label_clip_feats)

            logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
            logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])
            prob2 = (1 - logits2[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1))
            prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1))

            #prob1 = fixed_smooth(prob1, 7)
            ap3.append(prob1.cpu().detach().numpy())
            if i == 0:
                ap1 = prob1
                ap2 = prob2
                #ap3 = prob3
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
    pre, rec, _ = precision_recall_curve(gt, np.repeat(ap1, 16))
    print("AUC1: ", ROC1, " AP1: ", AP1)
    print("AUC2: ", ROC2, " AP2:", AP2)
    dmap, iou = dmAP(element_logits2_stack, gtsegments, gtlabels, excludeNormal=False)
    averageMAP = 0
    for i in range(5):
        print('mAP@{0:.1f} ={1:.2f}%'.format(iou[i], dmap[i]))
        averageMAP += dmap[i]
    averageMAP = averageMAP/(i+1)
    print('average MAP: {:.2f}'.format(averageMAP))
    return ROC1, AP1,averageMAP,dmap,ROC2


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option.parser.parse_args()

    label_map = dict({'Normal': 'Normal', 'Abuse': 'Abuse', 'Arrest': 'Arrest', 'Arson': 'Arson', 'Assault': 'Assault', 'Burglary': 'Burglary', 'Explosion': 'Explosion', 'Fighting': 'Fighting', 'RoadAccidents': 'RoadAccidents', 'Robbery': 'Robbery', 'Shooting': 'Shooting', 'Shoplifting': 'Shoplifting', 'Stealing': 'Stealing', 'Vandalism': 'Vandalism'})

    testdataset = UCFDataset(args.visual_length, args.test_list, True, label_map)
    testdataloader = DataLoader(testdataset, batch_size=1, shuffle=False)

    prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)
    model = SemVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device)
    model_param = torch.load(args.model_path)
    model.load_state_dict(model_param)
    patch_model = AttentionAcrossFrames(in_channels=512, grid_size=(7, 7)).to(device="cuda:0")
    patch_model.load_state_dict(
        torch.load("best_patch_model_0.41390380192494886_0.8584324820397643.pth", weights_only=True))
    test(model,patch_model, testdataloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)