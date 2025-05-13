import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from new_transformer_arch import  AttentionAcrossFrames
from model import CLIPVAD
from ucf_test import test
from utils.dataset import UCFDataset
from utils.tools import get_prompt_text, get_batch_label
import ucf_option
from torch.utils.data import DataLoader, RandomSampler
import pickle
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
def first_n_words(text, n):
    return " ".join(text.split()[:n])
def shuffle_dataloader(dataloader):
    dataloader.batch_sampler.sampler = RandomSampler(dataloader.dataset)
    return  dataloader
def CLASM(logits, labels, lengths, prompt_text,loaded_pkl,device,eps,enhenced_feats):
    normal = enhenced_feats[:int(enhenced_feats.shape[0] / 2), :, :]  # First 128 slices
    anomaly = enhenced_feats[int(enhenced_feats.shape[0] / 2):, :, :]
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)
    eps=0
    for i in range(logits.shape[0]):
        #torch.manual_seed(42)
        rand = torch.rand(1)

        if rand < eps and i>=logits.shape[0]/2:
            # Generate random indices
            #indices = torch.randperm(logits[i, 0:lengths[i]].size(0))[:int(lengths[i] / 16 + 1)]
            single_vector = anomaly[int(i - logits.shape[0] / 2)]

            # Normalize along the last dimension (dim=-1) to make the norm of each vector 1
            single_vector = F.normalize(single_vector, p=2, dim=-1)
            normal = F.normalize(normal, p=2, dim=-1)

            # Reshape the single vector to match the batch shape for broadcasting
            single_vector = single_vector.unsqueeze(0)  # Shape: (1, 1, 256, 512)

            # Compute Euclidean distance in a vectorized way
            distances = 1 - F.cosine_similarity(single_vector, normal,
                                                dim=-1)  # distances = torch.norm(single_vector - normal, dim=-1)  # Shape: (128, 256)
            distances = torch.mean(distances, dim=0)

            # print(distances)
            # Generate random indices
            val, _ = torch.topk(distances[0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
            # indices = torch.randperm(distances[i, 0:lengths[i]].size(0))[:int(lengths[i] / 16 + 1)]
            # Select values
            tmp = logits[i, 0:lengths[i]][_]
        else:
            tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)

    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    #torch.manual_seed(9)
    return milloss

def CLAS2(logits, labels, lengths, device,eps,enhenced_feats):

    normal = enhenced_feats[:int(enhenced_feats.shape[0]/2), :, :]  # First 128 slices
    anomaly = enhenced_feats[int(enhenced_feats.shape[0]/2):, :, :]
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])
    #torch.manual_seed(42)
    for i in range(logits.shape[0]):
        rand = torch.rand(1)
        if rand < eps and i>=logits.shape[0]/2:
            # Create the single vector (256, 512)
            single_vector = anomaly[int(i-logits.shape[0]/2)]

            # Normalize along the last dimension (dim=-1) to make the norm of each vector 1
            single_vector = F.normalize(single_vector, p=2, dim=-1)
            normal = F.normalize(normal, p=2, dim=-1)

            # Reshape the single vector to match the batch shape for broadcasting
            single_vector = single_vector.unsqueeze(0)  # Shape: (1, 1, 256, 512)

            # Compute Euclidean distance in a vectorized way
            distances = 1-F.cosine_similarity(single_vector, normal, dim=-1)#distances = torch.norm(single_vector - normal, dim=-1)  # Shape: (128, 256)
            distances = torch.mean(distances,dim=0)

            #print(distances)
            # Generate random indices
            val, _ = torch.topk(distances[0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
            #indices = torch.randperm(distances[i, 0:lengths[i]].size(0))[:int(lengths[i] / 16 + 1)]
            # Select values
            tmp = logits[i, 0:lengths[i]][_]

        else:
            tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)

        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat([instance_logits, tmp], dim=0)

    clsloss = F.binary_cross_entropy(instance_logits, labels)
    #torch.manual_seed(9)
    return clsloss

def train(model, normal_loader, anomaly_loader, testloader, args, label_map,label_map_t, device):
    model.to(device)
    #patch_model.to(device)
    #patch_model.train()
    break_counter = 0
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)
    with open("ucf_clip_labels.pkl", "rb") as f:
        loaded_data = pickle.load(f)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    prompt_text = get_prompt_text(label_map)
    prompt_text_t = get_prompt_text(label_map_t)
    ap_best = 0
    epoch = 0

    if args.use_checkpoint == True:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        ap_best = checkpoint['ap']
        print("checkpoint info:")
        print("epoch:", epoch+1, " ap:", ap_best)
    eps =0
    performance = []
    iterations = []
    for e in range(args.max_epoch):
        model.train()
        loss_total1 = 0
        loss_total2 = 0
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        for i in range(min(len(normal_loader), len(anomaly_loader))):
            step = 0
            normal_features, normal_label, normal_lengths,v_label_n = next(normal_iter)
            anomaly_features, anomaly_label, anomaly_lengths,v_label_a = next(anomaly_iter)
            a_stack = []
            for label_a in v_label_a:
                clip_feature_a = np.load("clip_256_ucf/"+label_a+".npy",allow_pickle=True)
                clip_feature_a = process_matrix(clip_feature_a)
                a_stack.append(clip_feature_a)
            a_stack = np.stack(a_stack).astype(np.float32)

            n_stack = []
            for label_n in v_label_n:
                clip_feature_a = np.load("clip_256_ucf/" + label_n + ".npy", allow_pickle=True)
                clip_feature_a = process_matrix(clip_feature_a)
                n_stack.append(clip_feature_a)
            n_stack = np.stack(n_stack).astype(np.float32)
            n_stack = torch.tensor(n_stack)
            a_stack = torch.tensor(a_stack).float()
            T_features = torch.cat([n_stack, a_stack], dim=0).to(device)
            visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(device)
            text_labels = list(normal_label) + list(anomaly_label)
            feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)
            prompt_text = get_prompt_text(label_map)
            text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)
            label_clip_feats = []
            for prompt in prompt_text:
                if "road" in prompt:
                    prompt = "RoadAccidents"
                else:
                    prompt = prompt.capitalize()
                clip = loaded_data[prompt]
                label_clip_feats.append(clip)
            label_clip_feats = np.stack(label_clip_feats)
            label_clip_feats = torch.tensor(label_clip_feats).to(device)
            enhanced_feats,label_feats,T_feats,text_features, logits1, logits2 = model(visual_features, None, prompt_text, feat_lengths,T_features,label_clip_feats)
            #loss1
            eps = eps*0.99
            #if eps < 0.05:
                #eps = 0.05
            loss1 = CLAS2(logits1, text_labels, feat_lengths, device,eps,enhanced_feats)
            loss_total1 += loss1.item()
            #loss2
            loss2 = CLASM(logits2, text_labels, feat_lengths,prompt_text,loaded_data, device,eps,enhanced_feats)
            loss_total2 += loss2.item()
            #loss3
            loss3 = torch.zeros(1).to(device)
            text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            for j in range(1, text_features.shape[0]):
                text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                loss3 += torch.abs(text_feature_normal @ text_feature_abr)
            loss3 = loss3 / 13 * 1e-1#'''

            inverse_cosine_sim = 0
            for l,logit in enumerate(logits2):
                text_label = text_labels[l]
                max_index = torch.argmax(text_label)
                target = label_feats[max_index]
                logit = logit.softmax(dim=-1)
                class_probs = logit[:, max_index]
                top_k_values, top_k_indices = torch.topk(class_probs, int(feat_lengths[l]/16) +1, largest=True)

                # Select the top-K instances from the original tensor
                top_k_instances = T_feats[l][top_k_indices]
                for top in top_k_instances:
                    inverse_cosine_sim = inverse_cosine_sim + (1-F.cosine_similarity(top, target,dim=0))/len(top_k_instances)
            inverse_cosine_sim = inverse_cosine_sim/len(logits2)#'''
            loss = loss1 + loss2 +loss3+ inverse_cosine_sim

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += i * normal_loader.batch_size * 2
            if step % 1280== 0 and step != 0:
                print('epoch: ', e+1,"Best: ",ap_best, '| step: ', step, '| loss1: ', loss_total1 / (i+1), '| loss2: ', loss_total2 / (i+1), '| loss3: ', loss3.item(), '| loss4: ', inverse_cosine_sim)
                AUC, AP , avgMap, dmap,AUC2= test(model, testloader, args.visual_length, prompt_text_t, gt, gtsegments, gtlabels, device)
                AUC = max(AUC,AUC2)
                break_counter = break_counter + 1
                if AUC > ap_best:
                    break_counter = 0
                    ap_best = AUC
                    checkpoint = {
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ap': ap_best}
                    torch.save(checkpoint, args.checkpoint_path)
                print("EPS: ",eps, "break_counter: ",break_counter)
                performance.append(AUC)
                iterations.append((e)*min(len(normal_loader), len(anomaly_loader))* normal_loader.batch_size * 2+ i * normal_loader.batch_size * 2)
            if break_counter >=30:
                break

        scheduler.step()
        # Create plot
        plt.figure(figsize=(6, 4))
        plt.plot(iterations, performance, marker='o', linestyle='-')

        # Labels and title
        plt.xlabel("X Axis")
        plt.ylabel("Y Axis")
        plt.title("Plot of iterations vs performance")
        # Save figure
        plt.savefig("plot_"+ str(args.seed) + "_"+str(args.batch_size)+".png", dpi=300)
        torch.save(model.state_dict(), 'model/model_cur.pth')
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if break_counter >= 30:
            break
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
    args = ucf_option.parser.parse_args()

    seeds = [12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]#9,234,42,1,256,#9,234,42,1,256,512,7,72,34,100
    results = []
    for seed in seeds:
        args.seed = seed

        label_map = dict({'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson', 'Assault': 'assault', 'Burglary': 'burglary', 'Explosion': 'explosion', 'Fighting': 'fighting', 'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery', 'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing', 'Vandalism': 'vandalism'})

        normal_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, True)
        normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        anomaly_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, False)
        anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

        test_dataset = UCFDataset(args.visual_length, args.test_list, True, label_map)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        setup_seed(args.seed)
        model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device)
        label_map_t = dict({'Normal': 'Normal', 'Abuse': 'Abuse', 'Arrest': 'Arrest', 'Arson': 'Arson', 'Assault': 'Assault', 'Burglary': 'Burglary', 'Explosion': 'Explosion', 'Fighting': 'Fighting', 'RoadAccidents': 'RoadAccidents', 'Robbery': 'Robbery', 'Shooting': 'Shooting', 'Shoplifting': 'Shoplifting', 'Stealing': 'Stealing', 'Vandalism': 'Vandalism'})

        #setup_seed(9)
        train(model, normal_loader, anomaly_loader, test_loader, args, label_map,label_map_t, device)

        label_map = dict({'Normal': 'Normal', 'Abuse': 'Abuse', 'Arrest': 'Arrest', 'Arson': 'Arson', 'Assault': 'Assault',
                          'Burglary': 'Burglary', 'Explosion': 'Explosion', 'Fighting': 'Fighting',
                          'RoadAccidents': 'RoadAccidents', 'Robbery': 'Robbery', 'Shooting': 'Shooting',
                          'Shoplifting': 'Shoplifting', 'Stealing': 'Stealing', 'Vandalism': 'Vandalism'})

        testdataset = UCFDataset(args.visual_length, args.test_list, True, label_map)
        testdataloader = DataLoader(testdataset, batch_size=1, shuffle=False)

        prompt_text = get_prompt_text(label_map)
        gt = np.load(args.gt_path)
        gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
        gtlabels = np.load(args.gt_label_path, allow_pickle=True)
        model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head,
                        args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device)
        model_param = torch.load(args.model_path)
        model.load_state_dict(model_param)
        AUC,AP,AvgMap,dmap,AUC2=test(model,testdataloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
        AUC = max(AUC,AUC2)
        results.append([AUC,AP,AvgMap,dmap[0],dmap[1],dmap[2],dmap[3],dmap[4],seed])
        df = pd.DataFrame(results)
        df.to_csv("seed_opt2.csv",header=None,index=False)