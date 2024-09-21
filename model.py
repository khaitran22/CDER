import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import dgl
from dgl import function as fn
from dgl.nn.functional import edge_softmax
from dgl.base import DGLError
from torch.nn import Identity
from dgl.utils import expand_as_pair

from collections import defaultdict
from long_seq import process_long_input
from opt_einsum import contract
from losses import SigmoidFocalLoss


def xavier(param):
    nn.init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm='both',
                 weight=True,
                 bias=True,
                 activation=None,
                 allow_zero_in_degree=False):
        super(GCN, self).__init__()
        if norm not in ('none', 'both', 'right', 'left'):
            raise DGLError('Invalid norm value. Must be either "none", "both", "right" or "left".'
                           ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree

        if weight:
            self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, weight=None, edge_weight=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')
            aggregate_fn = fn.copy_src('h', 'm')
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm in ['left', 'both']:
                degs = graph.out_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            if weight is not None:
                if self.weight is not None:
                    raise DGLError('External weight is provided while at the same time the'
                                   ' module has defined its own weight parameter. Please'
                                   ' create the module with flag weight=False.')
            else:
                weight = self.weight

            if self._in_feats > self._out_feats:
                # mult W first to reduce the feature size for aggregation.
                if weight is not None:
                    feat_src = torch.matmul(feat_src, weight)
                graph.srcdata['h'] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
            else:
                # aggregate first then mult W
                graph.srcdata['h'] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
                if weight is not None:
                    rst = torch.matmul(rst, weight)

            if self._norm in ['right', 'both']:
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst


class EP_GAT_PS(nn.Module):
    def __init__(self,
                in_feats,
                out_feats,
                num_heads,
                feat_drop=0.,
                attn_drop=0.,
                negative_slope=0.2,
                residual=False,
                activation=None,
                bias=True) -> None:
        
        super(EP_GAT_PS, self).__init__()
        self._num_heads = num_heads
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        
        self.fc = nn.Linear(self._in_feats, out_feats * num_heads, bias=False)
        self.fc_src = nn.Linear(self._in_feats, out_feats * num_heads, bias=False)
        self.fc_dst = nn.Linear(self._out_feats, out_feats * num_heads, bias=False)

        # Pair-sent
        self.attn_l_ps = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r_ps = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))

        # Sent-pair
        self.attn_l_sp = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r_sp = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))

        if bias:
            self.bias_sent = nn.Parameter(torch.FloatTensor(size=(num_heads * out_feats,)))
            self.bias_pair = nn.Parameter(torch.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        
        if residual:
            self.res_fc = nn.Linear(self._in_feats, num_heads * out_feats, bias=False) if self._in_feats != out_feats * num_heads else Identity()
        else:
            self.register_buffer('res_fc', None)
        
        self.reset_parameters()
        self.activation = activation
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)

        nn.init.xavier_normal_(self.attn_l_ps, gain=gain)
        nn.init.xavier_normal_(self.attn_r_ps, gain=gain)

        nn.init.xavier_normal_(self.attn_l_sp, gain=gain)
        nn.init.xavier_normal_(self.attn_r_sp, gain=gain)
        
        nn.init.constant_(self.bias_sent, 0)
        nn.init.constant_(self.bias_pair, 0)
        
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, graph, feats):
        edata_dict = defaultdict()

        with graph.local_scope():
            etypes = []
            for srctype, etype, dsttype in graph.canonical_etypes:
                rel_contxts, h_src, h_dst = feats[(srctype, etype, dsttype)]

                if etype == 'sp':
                    rel_contxts, h_sent, h_pair = feats[(srctype, etype, dsttype)]
                    h_src = h_sent
                    h_dst = rel_contxts
                else:
                    rel_contxts, h_pair, h_sent = feats[(srctype, etype, dsttype)]
                    h_src = rel_contxts
                    h_dst = h_sent
                
                src_prefix_shape = h_src.shape[:-1]
                dst_prefix_shape = h_dst.shape[:-1]

                feat_src = self.fc_src(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(*dst_prefix_shape, self._num_heads, self._out_feats)

                param_l, param_r = (self.attn_l_ps, self.attn_r_ps) if etype == 'ps' else (self.attn_l_sp, self.attn_r_sp)
                el = (feat_src * param_l).sum(dim=-1).unsqueeze(-1)
                er = (feat_dst * param_r).sum(dim=-1).unsqueeze(-1)

                if etype == 'sp':
                    graph.nodes[srctype].data.update({'el': el, 'ft': h_sent})
                    graph.nodes[dsttype].data.update({'er': er, 'ft': h_pair})
                else:
                    graph.nodes[srctype].data.update({'el': el, 'ft': h_pair})
                    graph.nodes[dsttype].data.update({'er': er, 'ft': h_sent})

                # Compute edge attention
                graph.apply_edges(
                    fn.u_dot_v('el', 'er', 'e'), 
                    etype=etype
                )

                e = graph.edges[etype].data.pop('e')

                # save data to compute probability later
                edata_dict.update({(srctype, etype, dsttype): e})

                etypes.append(etype)
        
            # Compute edge softmax
            e_attn = edge_softmax(graph, edata_dict)
            for k, v in e_attn.items():
                etype = k[1]
                graph.edges[etype].data['a'] = v
            
            update_funcs = {etype: (fn.v_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft_tmp')) for etype in etypes}
            graph.multi_update_all(update_funcs, 'sum')
            
            h_sent_tmp = graph.nodes['sent'].data['ft_tmp'] + self.bias_sent.view(*((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)
            h_pair_tmp = graph.nodes['pair'].data['ft_tmp'] + self.bias_pair.view(*((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)

            h_sent_tmp = h_sent_tmp.mean(dim=1)
            h_pair_tmp = h_pair_tmp.mean(dim=1)

            return h_pair_tmp, h_sent_tmp


class EP_GAT_PP(nn.Module):
    def __init__(self,
                in_feats,
                out_feats,
                num_heads,
                feat_drop=0.,
                attn_drop=0.,
                negative_slope=0.2,
                residual=False,
                activation=None,
                bias=True) -> None:
        
        super(EP_GAT_PP, self).__init__()
        self._num_heads = num_heads
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        
        self.fc = nn.Linear(self._in_feats, out_feats * num_heads, bias=False)

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        
        self.reset_parameters()
        self.activation = activation
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.constant_(self.bias, 0)

    def forward(self, graph, feats):
        with graph.local_scope():
            src_prefix_shape = dst_prefix_shape = feats[0].shape[:-1]
            # dst_prefix_shape = feats[0].shape[:-1]
            ft, e_ft = feats
            el = er = self.fc(e_ft).view(
                *src_prefix_shape, self._num_heads, self._out_feats
            )
            # el = er = e_ft

            graph.srcdata.update({"ft": ft, "el": el})
            graph.dstdata.update({"er": er})

            # Compute edge attention
            graph.apply_edges(
                fn.u_dot_v('el', 'er', 'e')
            )
            e = self.leaky_relu(graph.edata.pop("e"))
            
            # compute softmax
            graph.edata["a"] = edge_softmax(graph, e)
            
            graph.update_all(fn.v_mul_e("ft", "a", "m"), fn.sum("m", "ft_tmp"))
            rst = graph.dstdata["ft_tmp"]

            rst = rst + self.bias.view(*((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)
            rst = rst.mean(1)

            return rst


class CDER(nn.Module):
    def __init__(self, config, transformer, emb_size=768, max_sent_num=30, block_size=64) -> None:
        super().__init__()
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.encoder = transformer
        self.heads = [3, 3, 3]
        self.num_layers = 2
        self.max_sent_num = max_sent_num
        self.block_size = block_size
        self.emb_size = emb_size

        self.activation = nn.LeakyReLU(0.2)
  
        self.GCN_SS_Layer = nn.ModuleList()
        for i in range(self.num_layers):
            self.GCN_SS_Layer.append(
                GCN(self.hidden_size, self.emb_size, activation=self.activation)
            )

        self.GAT_PP_Layer = nn.ModuleList()
        for i in range(self.num_layers):
            self.GAT_PP_Layer.append(
                EP_GAT_PP(self.hidden_size, self.emb_size, num_heads=3, bias=True, activation=self.activation)
            )

        self.GAT_PS_Layer = nn.ModuleList()
        for i in range(self.num_layers):
            self.GAT_PS_Layer.append(
                EP_GAT_PS(self.emb_size, self.emb_size, num_heads=3, bias=True, activation=self.activation)
            )
        
        self.entity_pair_extractor = nn.Linear(3 * self.hidden_size, self.emb_size, bias=True)
        self.entity_pair_extractor.apply(weights_init)

        self.sentence_predictor = nn.Linear(self.emb_size * self.block_size, 1, bias=True)
        self.sentence_predictor.apply(weights_init)

        self.loss_fnt = SigmoidFocalLoss()

    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.encoder, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    def get_entity_embs(self, sequence_output, attention, entity_pos):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        entities_embs, entities_atts = [], []
        
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            for e in entity_pos[i]:
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]
            
            entities_embs.append(entity_embs)
            entities_atts.append(entity_atts)

        return entities_embs, entities_atts

    def get_sentence_embs(self, sequence_output, sent_maps):
        sents_embs = []
        for i in range(len(sent_maps)):
            current_seq = sequence_output.select(0, i)
            sent_embs = []
            current_sent_map = sent_maps[i]
            for sent_pos in current_sent_map:
                index = torch.arange(sent_pos[0], sent_pos[1]+1).to(current_seq.device)
                current_sent_embs = current_seq.index_select(0, index).mean(dim=0)
                sent_embs.append(current_sent_embs)
            sent_embs = torch.stack(sent_embs)
            sents_embs.append(sent_embs)

        return sents_embs
    
    def procs_entity_embs(self, hts, labels, entities_embs, entities_attns, report=False):
        hss, tss = [], []
        ht_attns = []
        epair_len = []
        for i in range(len(entities_embs)):
            entitiy_embs = entities_embs[i]
            entity_attns = entities_attns[i]
            current_hts = hts[i]

            if not report:
                current_label = torch.stack([torch.tensor(l) for l in labels[i]])[:, 1:]
                epair_idx = current_label.sum(dim=1).nonzero().squeeze(1).tolist()
                epair_len.append(len(epair_idx))
                current_hts = torch.stack([torch.tensor(current_hts[i]) for i in epair_idx]).to(entitiy_embs.device)
            else:
                current_hts = torch.stack([torch.tensor(current_hts[i]) for i in range(len(current_hts))]).to(entitiy_embs.device)
                epair_len.append(len(current_hts))
            
            hs = torch.index_select(entitiy_embs, 0, current_hts[:, 0])
            ts = torch.index_select(entitiy_embs, 0, current_hts[:, 1])
            h_attn = torch.index_select(entity_attns, 0, current_hts[:, 0])
            t_attn = torch.index_select(entity_attns, 0, current_hts[:, 1])
            ht_attn = (h_attn * t_attn).mean(1)
            ht_attn = ht_attn / (ht_attn.sum(1, keepdim=True) + 1e-5)
            ht_attns.append(ht_attn)
            hss.append(hs)
            tss.append(ts)
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)

        output = (hss, tss, epair_len, ht_attns)
        return output
    
    def create_mask(self, s_logits, sent_map, procs_epair_len):
        mask = [[] for i in range(s_logits.shape[0])]
        start = 0
        for i in range(len(sent_map)):
            doc_num_sent = sent_map[i]
            ent_pair_idx = torch.arange(start, start+procs_epair_len[i]).tolist()
            sent_idx = torch.arange(0, len(doc_num_sent)).tolist()
            for k in ent_pair_idx:
                j = 0
                while j < self.max_sent_num:
                    if j in sent_idx:
                        mask[k].append(True)
                    else:
                        mask[k].append(False)
                    j += 1
            start += procs_epair_len[i]

        mask = torch.stack([torch.tensor(m) for m in mask], dim=0).to(s_logits.device)

        return mask

    def get_doc_embs(self, h_sents, procs_epair_len, doc_idx):
        doc_sent_embs = []
        for i in range(len(doc_idx)):
            doc_sent_emb = torch.index_select(h_sents, 0, doc_idx[i].to(h_sents.device))
            doc_sent_emb = F.pad(doc_sent_emb, (0, 0, 0, self.max_sent_num - doc_sent_emb.shape[0]))
            doc_sent_embs.append(doc_sent_emb)
        
        doc_sent_reps = [doc_sent_embs[i].unsqueeze(0).expand(procs_epair_len[i], self.max_sent_num, self.emb_size) for i in range(len(procs_epair_len))]   
        doc_sent_reps = torch.cat(doc_sent_reps, dim=0)
        return doc_sent_reps

    def get_evi_labels(self, pair_evidence, h_pair_num):
        batch_pair_evidence = []
        for i in range(len(pair_evidence)):
            batch_pair_evidence.extend(pair_evidence[i])

        batch_pair_evidence_labels = torch.zeros((h_pair_num, self.max_sent_num))
        for i in range(len(batch_pair_evidence)):
            for j in batch_pair_evidence[i]:
                batch_pair_evidence_labels[i, j] = 1

        return batch_pair_evidence_labels

    def cosine_pairwise(self, x):
        x = x.permute((1, 2, 0))
        cos_sim_pairwise = F.cosine_similarity(x, x.unsqueeze(1), dim=-2)
        cos_sim_pairwise = cos_sim_pairwise.permute((2, 0, 1))
        th = 0.5
        pos_result = cos_sim_pairwise.gt(th)
        neg_result = cos_sim_pairwise.lt(th)
        pos_result = pos_result.squeeze(0).nonzero()
        neg_result = neg_result.squeeze(0).nonzero()
        return pos_result, neg_result

    def upgrade_pair_side_graph(self, old_gpp_subgraphs, pair_rel_contxts, procs_epair_len):
        start = 0
        for i in range(len(procs_epair_len)):
            current_old_gpp_subgraph = old_gpp_subgraphs[i]
            l = procs_epair_len[i]
            idx = torch.arange(start, start + l).to(pair_rel_contxts.device)
            current_pair_contxts = torch.index_select(pair_rel_contxts, 0, idx)
            high_related_pair, low_related_pair = self.cosine_pairwise(current_pair_contxts.unsqueeze(0))
            high_related_pair, low_related_pair = high_related_pair.cpu(), low_related_pair.cpu()

            # Add edge
            not_has_edge = current_old_gpp_subgraph.has_edges_between(high_related_pair[:, 0], high_related_pair[:, 1]).int().eq(0).nonzero().squeeze(1)
            added_pairs = torch.index_select(high_related_pair, 0, not_has_edge)
            current_old_gpp_subgraph.add_edges(added_pairs[:, 0], added_pairs[:, 1])
            current_old_gpp_subgraph.add_edges(added_pairs[:, 1], added_pairs[:, 0])

            # Remove edge
            if low_related_pair.shape[0] > 0:
                has_edge = current_old_gpp_subgraph.has_edges_between(low_related_pair[:, 0], low_related_pair[:, 1]).int().nonzero().squeeze(1)
                removed_pairs = torch.index_select(low_related_pair, 0, has_edge)
                if removed_pairs.shape[0] > 0:
                    edge_ids = current_old_gpp_subgraph.edge_ids(removed_pairs[:, 0], removed_pairs[:, 1])
                    current_old_gpp_subgraph = dgl.remove_edges(current_old_gpp_subgraph, edge_ids)

            old_gpp_subgraphs[i] = current_old_gpp_subgraph

        return old_gpp_subgraphs

    def forward(self, inputs, report=False):
        # doc. enc & preproc.
        sequence_output, attention = self.encode(inputs['input_ids'], inputs['attention_mask'])
        entities_embs, entities_attns = self.get_entity_embs(sequence_output, attention, inputs['entity_pos'])
        h_sent = torch.cat(self.get_sentence_embs(sequence_output, inputs['sent_map']), dim=0)
        
        start = 0
        doc_idx = []
        for i in range(len(inputs['sent_map'])):
            doc_len = len(inputs['sent_map'][i])
            idx = torch.arange(start, start+doc_len)
            doc_idx.append(idx)
            start += doc_len

        # ent. pair rep.
        hss, tss, procs_epair_len, procs_entities_attns = self.procs_entity_embs(
            hts=inputs['hts'],
            labels=inputs['labels'],
            entities_embs=entities_embs,
            entities_attns=entities_attns,
            report=report
        )
        rs = [contract("ld,rl->rd", torch.select(sequence_output, 0, i), procs_entities_attns[i]) for i in range(len(procs_epair_len))]
        rs = torch.cat(rs, dim=0)
        h_pair = torch.tanh(self.entity_pair_extractor(torch.cat([hss, tss, rs], dim=1)))

        # graph infer.
        trans_E_rel_embeds = tss - hss

        gss_subgraphs = inputs['gss_subgraph']
        gss_subgraphs = dgl.batch(gss_subgraphs).to(h_sent.device)
        
        gps_subgraph = inputs['gps_subgraph']
        gps_subgraph = dgl.batch(gps_subgraph).to(h_sent.device)
        
        gpp_subgraphs = inputs['gpp_subgraph']
        gpp_subgraphs = self.upgrade_pair_side_graph(gpp_subgraphs, trans_E_rel_embeds, procs_epair_len)
        for i in range(len(gpp_subgraphs)):
            gpp_subgraphs[i] = dgl.add_self_loop(gpp_subgraphs[i])
        gpp_subgraphs = dgl.batch(gpp_subgraphs).to(h_pair.device)

        h_sents, h_pairs = [h_sent], [h_pair]
        for i in range(self.num_layers):
            gcn_ss_layer = self.GCN_SS_Layer[i]
            gat_pp_layer = self.GAT_PP_Layer[i]
            gat_ps_layer = self.GAT_PS_Layer[i]
            
            h_sent_tmp_1 = gcn_ss_layer(gss_subgraphs, h_sent)
            h_pair_tmp_1 = gat_pp_layer(gpp_subgraphs, [h_pair, trans_E_rel_embeds])
            
            ps_feats = {
                ('pair', 'ps', 'sent'): (rs, h_pair, h_sent),
                ('sent', 'sp', 'pair'): (rs, h_sent, h_pair)
            }
            h_pair_tmp_2, h_sent_tmp_2 = gat_ps_layer(gps_subgraph, ps_feats)
            
            h_sent = torch.stack([h_sent, h_sent_tmp_1, h_sent_tmp_2], dim=0)
            h_pair = torch.stack([h_pair, h_pair_tmp_1, h_pair_tmp_2], dim=0)

            h_sent = self.activation(h_sent).mean(dim=0)
            h_pair = self.activation(h_pair).mean(dim=0)
            
            h_sents.append(h_sent)
            h_pairs.append(h_pair)
        
        h_pairs = torch.stack(h_pairs, dim=0).mean(dim=0)
        h_sents = torch.stack(h_sents, dim=0).mean(dim=0)

        # evidence prediction
        evi_labels = self.get_evi_labels(inputs['pair_evidence'], h_pairs.shape[0])
        s_inputs = self.get_doc_embs(h_sents, procs_epair_len, doc_idx).view(-1, self.max_sent_num, self.hidden_size // self.block_size, self.block_size)
        h_pairs = h_pairs.view(-1, self.hidden_size // self.block_size, self.block_size)
        s_inputs = (s_inputs.unsqueeze(4) * h_pairs.unsqueeze(1).unsqueeze(3)).view(-1, self.max_sent_num, self.hidden_size * self.block_size)
        s_logits = self.sentence_predictor(s_inputs).squeeze(-1)
        mask = self.create_mask(s_logits, inputs['sent_map'], procs_epair_len)
        s_logits = torch.where(mask, s_logits, torch.FloatTensor([-1e7]).to(s_logits.device))

        evi_loss = self.loss_fnt(s_logits, evi_labels.to(s_logits.device))
        output = (evi_loss, s_logits, evi_labels)

        return output