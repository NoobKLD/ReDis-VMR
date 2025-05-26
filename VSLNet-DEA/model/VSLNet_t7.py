import torch
import torch.nn as nn
from model.layers_t7 import Embedding, VisualProjection, FeatureEncoder, CQAttention, CQConcatenate, \
    ConditionedPredictor, HighLightLayer
from transformers import AdamW, get_linear_schedule_with_warmup


def build_optimizer_and_scheduler(model, configs):
    no_decay = ['bias', 'layer_norm', 'LayerNorm']  # no decay for parameters of layer norm and bias
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=configs.init_lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, configs.num_train_steps * configs.warmup_proportion,
                                                configs.num_train_steps)
    return optimizer, scheduler


class VSLNet(nn.Module):
    def __init__(self, configs, word_vectors):
        super(VSLNet, self).__init__()
        self.configs = configs
        self.word_vectors = word_vectors
        self.embedding_net = Embedding(num_words=configs.word_size, num_chars=configs.char_size, out_dim=configs.dim,
                                       word_dim=configs.word_dim, char_dim=configs.char_dim, word_vectors=word_vectors,
                                       drop_rate=configs.drop_rate)
        self.video_affine = VisualProjection(visual_dim=configs.video_feature_dim, dim=configs.dim,
                                             drop_rate=configs.drop_rate)
        self.feature_encoder = FeatureEncoder(dim=configs.dim, num_heads=configs.num_heads, kernel_size=7, num_layers=4,
                                              max_pos_len=configs.max_pos_len, drop_rate=configs.drop_rate)
        # video and query fusion
        self.cq_attention = CQAttention(dim=configs.dim, drop_rate=configs.drop_rate)
        self.cq_concat = CQConcatenate(dim=configs.dim)
        # query-guided highlighting
        self.highlight_layer = HighLightLayer(dim=configs.dim)
        # conditioned predictor
        self.predictor = ConditionedPredictor(dim=configs.dim, num_heads=configs.num_heads, drop_rate=configs.drop_rate,
                                              max_pos_len=configs.max_pos_len, predictor=configs.predictor)

        self.kld_loss = nn.KLDivLoss(reduction='mean', log_target=False)

        # init parameters
        self.init_parameters()

        self.dynamic = False

    def init_parameters(self):
        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                m.reset_parameters()
        self.apply(init_weights)

    def forward(self, word_ids, char_ids, video_features, v_mask, q_mask):
        query_features = self.embedding_net(word_ids, char_ids)
        if not self.dynamic:
            video_features = self.video_affine(video_features)
            video_features = self.feature_encoder(video_features, mask=v_mask)
            query_features = self.feature_encoder(query_features, mask=q_mask)
            features = self.cq_attention(video_features, query_features, v_mask, q_mask)
            features = self.cq_concat(features, query_features, q_mask)
            h_score = self.highlight_layer(features, v_mask)
            features = features * h_score.unsqueeze(2)
        else:
            video_features2 = self.video_affine2(video_features)
            video_features2 = self.feature_encoder2(video_features2, mask=v_mask)
            query_features2 = self.feature_encoder2(query_features, mask=q_mask)
            features2 = self.cq_attention2(video_features2, query_features2, v_mask, q_mask)
            features2 = self.cq_concat(features2, query_features2, q_mask)
            h_score2 = self.highlight_layer2(features2, v_mask)

            video_features = self.video_affine(video_features)
            video_features = self.feature_encoder(video_features, mask=v_mask)
            query_features = self.feature_encoder(query_features, mask=q_mask)
            features = self.cq_attention(video_features, query_features, v_mask, q_mask)
            features = self.cq_concat(features, query_features, q_mask)
            h_score = self.highlight_layer(features, v_mask)

            features = torch.concatenate([
                features * h_score.unsqueeze(2),
                features2 * h_score2.unsqueeze(2),
            ], dim=-1)
            features = self.linear(features)

        start_logits, end_logits = self.predictor(features, mask=v_mask)
        return h_score, start_logits, end_logits

    def dynamic_model(self, device):
        # dynamic expand model
        self.video_affine2 = VisualProjection(visual_dim=self.configs.video_feature_dim, dim=self.configs.dim,
                                             drop_rate=self.configs.drop_rate).to(device)
        self.feature_encoder2 = FeatureEncoder(dim=self.configs.dim, num_heads=self.configs.num_heads, kernel_size=7, num_layers=4,
                                              max_pos_len=self.configs.max_pos_len, drop_rate=self.configs.drop_rate).to(device)
        # video and query fusion
        self.cq_attention2 = CQAttention(dim=self.configs.dim, drop_rate=self.configs.drop_rate).to(device)
        self.cq_concat2 = CQConcatenate(dim=self.configs.dim).to(device)
        # query-guided highlighting
        self.highlight_layer2 = HighLightLayer(dim=self.configs.dim).to(device)


        self.linear = nn.Linear(self.configs.dim * 2, self.configs.dim).to(device)

        # for par in self.embedding_net.parameters():
        #     par.requires_grad = False
        # for par in self.video_affine.parameters():
        #     par.requires_grad = False
        # for par in self.feature_encoder.parameters():
        #     par.requires_grad = False
        # for par in self.cq_concat.parameters():
        #     par.requires_grad = False
        # for par in self.cq_attention.parameters():
        #     par.requires_grad = False
        for par in self.predictor.parameters():
            par.requires_grad = False
        # for par in self.highlight_layer.parameters():
        #     par.requires_grad = False

        self.dynamic = True

    def extract_index(self, start_logits, end_logits):
        return self.predictor.extract_index(start_logits=start_logits, end_logits=end_logits)

    def compute_highlight_loss(self, scores, labels, mask):
        return self.highlight_layer.compute_loss(scores=scores, labels=labels, mask=mask)

    def compute_loss(self, start_logits, end_logits, start_labels, end_labels):
        return self.predictor.compute_cross_entropy_loss(start_logits=start_logits, end_logits=end_logits,
                                                         start_labels=start_labels, end_labels=end_labels)

    def compute_fair_loss(self, pre, tar):
        return self.kld_loss(input=pre, target=tar)
