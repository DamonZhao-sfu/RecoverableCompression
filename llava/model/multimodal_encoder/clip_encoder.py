import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig, CLIPTokenizer, CLIPTextModel, CLIPTextConfig, CLIPVisionModelWithProjection, CLIPTextModelWithProjection, CLIPModel

import matplotlib.pyplot as plt
import numpy as np
import json
import os
from sklearn.neighbors import LocalOutlierFactor
 
 
def complement_idx(idx, dim):
    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1, )
    for i in range(1, ndim):
        a = a.unsqueeze(0)
    a = a.expand(*dims)
    masked = torch.scatter(a, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl

outputs = {}
def hook_k(module, input, output):
    outputs['desired_k'] = output

def hook_q(module, input, output):
    outputs['desired_q'] = output

def find_max_derivative_position(data, k=10,contamination='auto'):
    data = data.to(dtype=torch.float32).cpu().numpy().flatten()  
    data = [[data[i]] for i in range(len(data))]
    
    lof = LocalOutlierFactor(n_neighbors=k, contamination=contamination)  
    y_pred = lof.fit_predict(data)  
    
    outlier_count = np.sum(y_pred == -1)  
    outlier_ratio = outlier_count / len(y_pred)  
      
    return outlier_ratio    


class CLIPTextEncoder(nn.Module):
    def __init__(self, text_encoder, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.tokenzier_llama = AutoTokenizer.from_pretrained("./vicuna-7b-v1.5", use_fast=False)
        self.text_encoder_name = text_encoder
        self.select_layer = args.mm_vision_select_layer 
        self.select_feature = getattr(args, 'mm_text_select_feature', 'patch')
        self.max_length     = args.max_length
        
        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPTextConfig.from_pretrained(self.text_encoder_name)
            
    def load_model(self):
        self.text_processor = CLIPTokenizer.from_pretrained(self.text_encoder_name)
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(self.text_encoder_name) 
        self.text_encoder.requires_grad_(False)

        self.is_loaded = True
        
    def feature_select(self, text_forward_outs):
        text_features = text_forward_outs.hidden_states[self.select_layer] 
        if self.select_feature == 'patch':
            text_features = text_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            text_features = text_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return text_features
    
    @torch.no_grad()
    def forward(self, input_ids_noim):
        input_text = self.tokenzier_llama.decode(input_ids_noim) 
        input_text = input_text.split('\n')
        
        input_text_ = input_text[0]
        
        input_ids = self.text_processor(input_text_, max_length=self.max_length , truncation=True, padding="max_length", return_tensors="pt")      

        attention_mask = input_ids['attention_mask'].data
        attention_mask = attention_mask.bool()

        text_forward_out = self.text_encoder(input_ids["input_ids"].to(device=self.device))
        
        text_forward_out = text_forward_out['last_hidden_state']
        text_features = text_forward_out[:,-1,:]
        text_features = self.text_encoder.text_projection(text_features)
      
        return (text_features, input_text_)
    
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.text_encoder.dtype

    @property
    def device(self):
        return self.text_encoder.device

    @property
    def config(self):
        if self.is_loaded:
            return self.text_encoder.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    
class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer # default: -2
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.total_tokens = 0
        
        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower_with_project = CLIPVisionModelWithProjection.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)
        self.vision_tower_with_project.requires_grad_(False)
        
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer] # penultimate layer output
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def read_json(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    
    def get_score_by_text(self, images, text_feature):
        
        input_text = text_feature[-1]
        text_feature = text_feature[0]
        
        images_features = self.vision_tower_with_project(images.to(device=self.device, dtype=self.dtype))
        images_features_output_project = self.vision_tower_with_project.visual_projection(images_features['last_hidden_state'])  # [B, 576+1, 768]
        
        text_features = text_feature.to(images_features_output_project.device) # [B, 768]    
            
        image_features = images_features_output_project[:,1:,:]                                                # [B, 576, 768]
        image_cls = images_features_output_project[:,0,:]                                                      # [B, 768]
        
        # Normalize
        text_features = F.normalize(text_features, p=2, dim=-1)
        image_features_ = image_features
        image_features__ = F.normalize(image_features, p=2, dim=-1)
        
        # # Calculate similarity between the vision and the text ----- Image-text score
        similarity = text_features @ image_features__.transpose(-2, -1) * image_features.shape[-1] ** -0.5
        similarity = similarity
            
        return (similarity, input_text)
        
    
    def token_recycling_with_text_clustering(self, images, text_score, reduction_ratio = 0.05):

        hook_handle_k = self.vision_tower.vision_model.encoder.layers[23].self_attn.k_proj.register_forward_hook(hook_k)
        hook_handle_q = self.vision_tower.vision_model.encoder.layers[23].self_attn.q_proj.register_forward_hook(hook_q)
        
        similarity = text_score[0]
        input_text = text_score[1]
        # forward pass
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        cls_token_last_layer =image_forward_outs.hidden_states[self.select_layer][:, 0:1]
        image_features = self.feature_select(image_forward_outs).to(images.dtype)          
        
        
        B, N, C = image_features.shape

        #extract desired layer's k and q and remove hooks; calculate attention
        desired_layer_k = outputs["desired_k"]
        desired_layer_q = outputs["desired_q"]

        hook_handle_k.remove()
        hook_handle_q.remove()

        attn = (desired_layer_q @ desired_layer_k.transpose(-2, -1)) * C ** -0.5
        attn = F.softmax(attn, dim=-1)

        cls_attn = attn[:, 0, 1:]  

        reduction_ratio_vision = find_max_derivative_position(cls_attn)
        
        _, index_by_vision = torch.topk(cls_attn, int(cls_attn.shape[-1]*reduction_ratio_vision), dim=1, largest=True)
        
        reduction_ratio_text = find_max_derivative_position(similarity[:,0,:])
        
        _, index_by_text = torch.topk(similarity[:,0,:], int(similarity.shape[-1]*reduction_ratio_text), dim=1, largest=True)
        
        index_concat = torch.cat((index_by_vision, index_by_text), dim=-1)     
        index_union = torch.unique(index_concat)
        index_intersect = np.intersect1d(index_by_vision.cpu().numpy(), index_by_text.cpu().numpy())

        index = index_union.unsqueeze(0).unsqueeze(-1).expand(-1, -1, C)
        image_features_main = torch.gather(image_features, dim=1, index=index)
        
        
        index_other = complement_idx(index_union.unsqueeze(0), N)
        cls_attn = torch.gather(cls_attn, dim=1, index=index_other)
        
        reduction_ratio = find_max_derivative_position(cls_attn)
        _, idx = torch.topk(cls_attn, int(N*reduction_ratio), dim=1, largest=True)  # [B, left_tokens] , sorted=True
        index_ = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]
        
        image_features = torch.gather(image_features, dim=1, index=index_other.unsqueeze(-1).expand(-1, -1, C))
        
        B, N, C = image_features.shape
        
        image_text_score = torch.gather(similarity[:,0,:], dim=1, index=index_other)
        image_text_score = 50*image_text_score
        
        Key_wo_cls = desired_layer_k[:, 1:]  # [B, N-1, C]
        
        x_others = torch.gather(image_features, dim=1, index=index_)  # [B, left_tokens, C] cluster centers
        x_others_attn = torch.gather(cls_attn, dim=1, index=idx)  
        x_other_text_score = torch.gather(image_text_score, dim=1, index=idx)
        Key_others = torch.gather(Key_wo_cls, dim=1, index=index_)  # [B, left_tokens, C]
        
        compl = complement_idx(idx, N)  # [B, N-1-left_tokens]
        non_topk = torch.gather(image_features, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))  # [B, N-1-left_tokens, C]
        non_topk_Key = torch.gather(Key_wo_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))
        non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)  # [B, N-1-left_tokens]
        non_topk_text_score = torch.gather(image_text_score, dim=1, index=compl)
        
        Key_others_norm = F.normalize(Key_others, p=2, dim=-1)
        non_topk_Key_norm = F.normalize(non_topk_Key, p=2, dim=-1)

        B, left_tokens, C = x_others.size()
        updated_x_others = torch.zeros_like(x_others)

        for b in range(B):
            for i in range(left_tokens):
                key_others_norm = Key_others_norm[b,i,:].unsqueeze(0).unsqueeze(0)
                
                before_i_Key = Key_others_norm[b, :i, :].unsqueeze(0)  
                after_i_Key = Key_others_norm[b, i+1:, :].unsqueeze(0) 
                
                before_i_x_others = x_others[b, :i, :].unsqueeze(0)  
                after_i_x_others = x_others[b, i+1:, :].unsqueeze(0)   
                rest_x_others = torch.cat([before_i_x_others, after_i_x_others, non_topk[b,:,:].unsqueeze(0)], dim=1)   
                before_i_x_others_attn = x_others_attn[b, :i].unsqueeze(0)  
                after_i_x_others_attn = x_others_attn[b, i+1:].unsqueeze(0)  
                rest_x_others_attn = torch.cat([before_i_x_others_attn, after_i_x_others_attn, non_topk_attn[b,:].unsqueeze(0)], dim=1)  

                rest_Keys = torch.cat([before_i_Key, after_i_Key, non_topk_Key_norm[b,:,:].unsqueeze(0)], dim=1)
                cos_sim_matrix = torch.bmm(key_others_norm, rest_Keys.transpose(1, 2))
                
                key_other_text_score = x_other_text_score[b,i].unsqueeze(0) 
                before_i_key_text_score = x_other_text_score[b, :i].unsqueeze(0) 
                after_i_kry_text_score = x_other_text_score[b, i+1:].unsqueeze(0) 
                
                rest_x_others_text_score = torch.cat([before_i_key_text_score, after_i_kry_text_score, non_topk_text_score[b,:].unsqueeze(0)], dim=1)
                _, cluster_indices = torch.topk((cos_sim_matrix + rest_x_others_text_score.unsqueeze(0) 
                ), k=int(32), dim=2, largest=True)

                cluster_tokens = rest_x_others[:,cluster_indices.squeeze(),:]
                
                # update cluster centers
                weights = rest_x_others_attn[:,cluster_indices.squeeze()].unsqueeze(-1) 
                        
                # update cluster centers
                weighted_avg = torch.sum(cluster_tokens * weights, dim=1) #/ torch.sum(weights)
                updated_center = weighted_avg + x_others[b, i, :]  
                updated_x_others[b, i, :] = updated_center 
            

        extra_one_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, C]
        updated_x_others = torch.cat([updated_x_others, extra_one_token],dim=1)
        image_features = updated_x_others
        image_features = torch.cat((image_features_main,image_features), dim=1)     
        return image_features
    
    def token_recycling_with_text_in_vision_and_clustring(self, images, text_score, reduction_ratio = 0.05):
        hook_handle_k = self.vision_tower.vision_model.encoder.layers[23].self_attn.k_proj.register_forward_hook(hook_k)
        hook_handle_q = self.vision_tower.vision_model.encoder.layers[23].self_attn.q_proj.register_forward_hook(hook_q)
        
        similarity = text_score[0]
        input_text = text_score[1]
        
        # forward pass
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        cls_token_last_layer =image_forward_outs.hidden_states[self.select_layer][:, 0:1]
        image_features = self.feature_select(image_forward_outs).to(images.dtype)  

        B, N, C = image_features.shape

        #extract desired layer's k and q and remove hooks; calculate attention
        desired_layer_k = outputs["desired_k"]
        desired_layer_q = outputs["desired_q"]

        hook_handle_k.remove()
        
        hook_handle_q.remove()

        attn = (desired_layer_q @ desired_layer_k.transpose(-2, -1)) * C ** -0.5
        attn = F.softmax(attn, dim=-1)

        cls_attn = attn[:, 0, 1:]  
        
        reduction_ratio_vision = find_max_derivative_position(cls_attn,k=20)
        
        _, index_by_vision = torch.topk(cls_attn, int(cls_attn.shape[-1]*reduction_ratio_vision), dim=1, largest=True)

        similarity = similarity[:,0,:]
        min_t,___ = torch.min(similarity,dim=-1)
        similarity[:,index_by_vision[0,:]] = min_t
        reduction_ratio_text = find_max_derivative_position(similarity,k=20)
        
        _, index_by_text = torch.topk(similarity, int((similarity.shape[-1] - index_by_vision.shape[-1])*reduction_ratio_text), dim=1, largest=True)
        index_all = torch.cat((index_by_vision, index_by_text), dim=-1)
        index_all_s,__ = torch.sort(index_all, dim=-1)
        
        index = index_all_s.unsqueeze(-1).expand(-1, -1, C)
        image_features_all = torch.gather(image_features, dim=1, index=index)
        
        index_other = complement_idx(index_all_s, N)
        N = index.shape[-1]
        sim = torch.gather(similarity, dim=1, index=index_other.squeeze(1))
        cls_attn = torch.gather(cls_attn, dim=1, index=index_other.squeeze(1))
        
        reduction_ratio = find_max_derivative_position(cls_attn)
        
        _, idx = torch.topk(cls_attn, int(N*reduction_ratio), dim=1, largest=True)  # [B, left_tokens] , sorted=True
        index_ = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]
        
        image_features = torch.gather(image_features, dim=1, index=index_other.unsqueeze(-1).expand(-1, -1, C))
        
        B, N, C = image_features.shape
        
        image_text_score = torch.gather(similarity, dim=1, index=index_other)
        image_text_score = 50*image_text_score
         
        Key_wo_cls = desired_layer_k[:, 1:]  # [B, N-1, C]
        
        x_others = torch.gather(image_features, dim=1, index=index_)  # [B, left_tokens, C] cluster centers
        x_others_attn = torch.gather(cls_attn, dim=1, index=idx)  
        x_other_text_score = torch.gather(image_text_score, dim=1, index=idx)
        Key_others = torch.gather(Key_wo_cls, dim=1, index=index_)  # [B, left_tokens, C]
        
        compl = complement_idx(idx, N)  # [B, N-1-left_tokens]
        non_topk = torch.gather(image_features, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))  # [B, N-1-left_tokens, C]
        non_topk_Key = torch.gather(Key_wo_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))
        non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)  # [B, N-1-left_tokens]
        non_topk_text_score = torch.gather(image_text_score, dim=1, index=compl)
        
        Key_others_norm = F.normalize(Key_others, p=2, dim=-1)
        non_topk_Key_norm = F.normalize(non_topk_Key, p=2, dim=-1)

        B, left_tokens, C = x_others.size()
        updated_x_others = torch.zeros_like(x_others)

        for b in range(B):
            for i in range(left_tokens):
                key_others_norm = Key_others_norm[b,i,:].unsqueeze(0).unsqueeze(0)
                
                before_i_Key = Key_others_norm[b, :i, :].unsqueeze(0)  
                after_i_Key = Key_others_norm[b, i+1:, :].unsqueeze(0) 
                
                before_i_x_others = x_others[b, :i, :].unsqueeze(0)  
                after_i_x_others = x_others[b, i+1:, :].unsqueeze(0)   
                rest_x_others = torch.cat([before_i_x_others, after_i_x_others, non_topk[b,:,:].unsqueeze(0)], dim=1)   
                before_i_x_others_attn = x_others_attn[b, :i].unsqueeze(0)  
                after_i_x_others_attn = x_others_attn[b, i+1:].unsqueeze(0)  
                rest_x_others_attn = torch.cat([before_i_x_others_attn, after_i_x_others_attn, non_topk_attn[b,:].unsqueeze(0)], dim=1)  

                rest_Keys = torch.cat([before_i_Key, after_i_Key, non_topk_Key_norm[b,:,:].unsqueeze(0)], dim=1)
                cos_sim_matrix = torch.bmm(key_others_norm, rest_Keys.transpose(1, 2))
                
                key_other_text_score = x_other_text_score[b,i].unsqueeze(0) 
                before_i_key_text_score = x_other_text_score[b, :i].unsqueeze(0) 
                after_i_kry_text_score = x_other_text_score[b, i+1:].unsqueeze(0) 
                
                rest_x_others_text_score = torch.cat([before_i_key_text_score, after_i_kry_text_score, non_topk_text_score[b,:].unsqueeze(0)], dim=1)
                _, cluster_indices = torch.topk((cos_sim_matrix + rest_x_others_text_score.unsqueeze(0) 
                ), k=int(32), dim=2, largest=True)

                cluster_tokens = rest_x_others[:,cluster_indices.squeeze(),:]
                
                # update cluster centers
                weights = rest_x_others_attn[:,cluster_indices.squeeze()].unsqueeze(-1) 
                        
                # update cluster centers
                weighted_avg = torch.sum(cluster_tokens * weights, dim=1) #/ torch.sum(weights)
                updated_center = weighted_avg + x_others[b, i, :]  
                updated_x_others[b, i, :] = updated_center 
            
        extra_one_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, C]
        updated_x_others = torch.cat([updated_x_others, extra_one_token],dim=1)
        image_features = updated_x_others
        
        if image_features.shape[1] != 1:
            image_features = torch.cat((image_features_all,image_features), dim=1)
        else:
            
            image_features = image_features_all
        return image_features
    
    @torch.no_grad()
    def forward(self, images, text_features):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            text_scroe = self.get_score_by_text(images, text_features)
            image_features = self.token_recycling_with_text_in_vision_and_clustring(images, text_scroe, reduction_ratio = 0.05)
            # image_features = self.token_recycling_with_text_clustering(images, text_scroe, reduction_ratio = 0.05)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2