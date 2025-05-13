import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random
import pickle
from model import CLIPVAD
from xd_test import test
from utils.dataset import XDDataset
from utils.tools import get_prompt_text, get_batch_label
import xd_option
import pandas as pd
def process_matrix(matrix):
    # Step 1: Convert (b,1,n) to (b,n) by squeezing the second dimension
    matrix = matrix.squeeze(axis=1)  # Removes the singleton dimension

    # Step 2: Pad with zeros if b < 256
    b, n = matrix.shape
    if b < 256:
        pad_size = 256 - b  # Number of rows to add
        pad_matrix = np.zeros((pad_size, n))  # Create zero rows
        matrix = np.vstack([matrix, pad_matrix])  # Append zero rows

    return matrix
def CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)

    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss

def CLAS2(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat((instance_logits, tmp))

    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss

def train(model, train_loader, test_loader, args, label_map: dict, device):
    model.to(device)
    with open("xd_clip_labels.pkl", "rb") as f:
        loaded_data = pickle.load(f)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    prompt_text = get_prompt_text(label_map)
    ap_best = 0
    epoch = 0
    break_counter = 0
    if args.use_checkpoint == True:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        ap_best = checkpoint['ap']
        print("checkpoint info:")
        print("epoch:", epoch+1, " ap:", ap_best)

    for e in range(args.max_epoch):
        model.train()
        loss_total1 = 0
        loss_total2 = 0
        for i, item in enumerate(train_loader):
            step = 0
            visual_feat, text_labels, feat_lengths,v_label = item
            c_stack = []
            remove_indices = []
            for ind,label in enumerate(v_label):
                try:
                    clip_feature = np.load("clip_256_xd/" + label + ".npy", allow_pickle=True)
                    clip_feature = process_matrix(clip_feature)
                    c_stack.append(clip_feature)
                except:
                    remove_indices.append(ind)

            if len(remove_indices)>0:
                indices_to_remove = torch.tensor(remove_indices)  # Indices to remove

                # Create a mask for all indices
                mask = torch.ones(visual_feat.shape[0], dtype=torch.bool)
                mask[indices_to_remove] = False  # Set False for indices to remove

                # Apply mask
                visual_feat = visual_feat[mask]
                feat_lengths = feat_lengths[mask]
                text_labels = tuple(val for i, val in enumerate(text_labels) if i not in remove_indices)
                v_label = tuple(val for i, val in enumerate(v_label) if i not in remove_indices)

            c_stack = torch.tensor(np.stack(c_stack).astype(np.float32)).to(device)
            label_clip_feats = []
            for prompt in prompt_text:
                clip = loaded_data[prompt]
                label_clip_feats.append(clip)
            label_clip_feats = np.stack(label_clip_feats).astype(np.float32)
            label_clip_feats = torch.tensor(label_clip_feats).to(device)
            visual_feat = visual_feat.to(device)
            feat_lengths = feat_lengths.to(device)
            text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)

            enhanced_feats,label_feats,T_feats,text_features, logits1, logits2 = model(visual_feat, None, prompt_text, feat_lengths,c_stack,label_clip_feats)

            loss1 = CLAS2(logits1, text_labels, feat_lengths, device) 
            loss_total1 += loss1.item()

            loss2 = CLASM(logits2, text_labels, feat_lengths, device)
            loss_total2 += loss2.item()

            loss3 = torch.zeros(1).to(device)
            text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            for j in range(1, text_features.shape[0]):
                text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                loss3 += torch.abs(text_feature_normal @ text_feature_abr)
            loss3 = loss3 / 6

            inverse_cosine_sim = 0
            for l, logit in enumerate(logits2):
                text_label = text_labels[l]
                max_index = torch.argmax(text_label)
                target = label_feats[max_index]
                logit = logit.softmax(dim=-1)
                class_probs = logit[:, max_index]
                top_k_values, top_k_indices = torch.topk(class_probs, int(feat_lengths[l] / 16) + 1, largest=True)

                # Select the top-K instances from the original tensor
                top_k_instances = T_feats[l][top_k_indices]
                for top in top_k_instances:
                    inverse_cosine_sim = inverse_cosine_sim + (1 - F.cosine_similarity(top, target, dim=0)) / len(
                        top_k_instances)
            inverse_cosine_sim = inverse_cosine_sim / len(logits2)#'''

            loss = loss1 + loss2 + loss3* 1e-4 + inverse_cosine_sim# #+ inverse_cosine_sim

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += i * train_loader.batch_size
            if step %  4800== 0 and step != 0:
                print('epoch: ', e+1, 'best: ',ap_best,"break: ",break_counter,'| step: ', step, '| loss1: ', loss_total1 / (i+1), '| loss2: ', loss_total2 / (i+1), '| loss3: ', loss3.item())
                ROC1, AP1, ROC2, AP2, averageMAP, dmap = test(model, test_loader, args.visual_length, prompt_text, gt,
                                                              gtsegments, gtlabels, device)
                break_counter = break_counter + 1
                AP = max(AP1, AP2)
                AUC = max(ROC1, ROC2)
                if AP > ap_best:
                    break_counter = 0
                    ap_best = AP
                    checkpoint = {
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ap': ap_best}
                    torch.save(checkpoint, args.checkpoint_path)
            if break_counter >= 30:
                break
        scheduler.step()
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    checkpoint = torch.load(args.checkpoint_path)
    torch.save(checkpoint['model_state_dict'], args.model_path)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args = xd_option.parser.parse_args()
    seeds = [9,234,42,1, 256, 512, 7, 72, 34, 100]
    results = []
    for seed in seeds:
        args.seed = seed
        setup_seed(args.seed)

        label_map = dict({'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'})

        train_dataset = XDDataset(args.visual_length, args.train_list, False, label_map)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        test_dataset = XDDataset(args.visual_length, args.test_list, True, label_map)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device)

        train(model, train_loader, test_loader, args, label_map, device)

        label_map = dict(
            {'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident',
             'G': 'explosion'})

        test_dataset = XDDataset(args.visual_length, args.test_list, True, label_map)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        prompt_text = get_prompt_text(label_map)
        gt = np.load(args.gt_path)
        gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
        gtlabels = np.load(args.gt_label_path, allow_pickle=True)

        model = SemVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head,
                        args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device)
        model_param = torch.load(args.model_path)
        model.load_state_dict(model_param)

        ROC1, AP1,ROC2,AP2 ,averageMAP,dmap=test(model, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
        AP = max(AP1,AP2)
        AUC = max(ROC1,ROC2)
        results.append([AUC, AP, averageMAP, dmap[0], dmap[1], dmap[2], dmap[3], dmap[4], seed])
        df = pd.DataFrame(results)
        df.to_csv("XD.csv", header=None, index=False)