import os

from tqdm import tqdm
import ujson as json
import dgl

docred_rel2id = json.load(open('./dataset/docred/meta/rel2id.json', 'r'))

def build_graphs(hts, raw_evidences, doc_length, entity_pos_sentence):
    # entity pair graph
    all_pos_pair = [l for l in range(len(hts))]
    src_pair_nodes, dst_pair_nodes = [], []
    added_edges = {}
    pair_used = []

    for i in range(len(hts)):
        for j in range(i+1, len(hts)):
            src_p, dst_p = set(hts[i]), set(hts[j])
            if added_edges.get((i, j)) == None and len(src_p & dst_p) > 0:
                src_pair_nodes.append(i)
                dst_pair_nodes.append(j)
                added_edges[(i, j)] = 1
                added_edges[(j, i)] = 1
                pair_used.extend([i, j])

    gpp_subgraph = dgl.graph((src_pair_nodes, dst_pair_nodes))
    distinct_pair_used = sorted(set(pair_used))

    if len(distinct_pair_used) > 0:
        if all_pos_pair[-1] > distinct_pair_used[-1]:
            gpp_subgraph = dgl.add_nodes(gpp_subgraph, all_pos_pair[-1] - distinct_pair_used[-1])
    else:
        gpp_subgraph = dgl.add_nodes(gpp_subgraph, len(all_pos_pair))

    gpp_subgraph = dgl.add_self_loop(gpp_subgraph)
    gpp_subgraph = dgl.to_bidirected(gpp_subgraph)
    
    # sentence graph
    all_sentences = [l for l in range(doc_length)]
    src_sent_nodes, dst_sent_nodes = [], []
    added_edges = {}
    sent_used = []

    for _, v in entity_pos_sentence.items():
        for i in range(len(v)):
            for j in range(i+1, len(v)):
                if added_edges.get((v[i], v[j])) == None:
                    src_sent_nodes.append(v[i])
                    dst_sent_nodes.append(v[j])
                    added_edges[(v[i], v[j])] = 1
                    added_edges[(v[j], v[i])] = 1
                    sent_used.extend([v[j], v[i]])

    gss_subgraph = dgl.graph((src_sent_nodes, dst_sent_nodes))
    
    # add missing nodes in the sentence graph
    distinct_sent_used = sorted(set(sent_used))
    if len(distinct_sent_used) > 0:
        if all_sentences[-1] > distinct_sent_used[-1]:
            gss_subgraph = dgl.add_nodes(gss_subgraph, all_sentences[-1] - distinct_sent_used[-1])
    else:
        gss_subgraph = dgl.add_nodes(gss_subgraph, len(all_sentences))

    gss_subgraph = dgl.add_self_loop(gss_subgraph)
    gss_subgraph = dgl.to_bidirected(gss_subgraph)

    # bipartite graph
    src_pair2sent_nodes, dst_pair2sent_nodes = [], []
    src_sent2pair_nodes, dst_sent2pair_nodes = [], []
    sent_used = []
    
    for i in range(len(hts)):
        raw_evidence = raw_evidences[i]
        for es in raw_evidence:
            src_pair2sent_nodes.append(i)
            dst_pair2sent_nodes.append(es)
            sent_used.append(es)
    
    for s in all_sentences:
        for i in range(len(hts)):
            current_raw_evidences = raw_evidences[i]
            if s in current_raw_evidences:
                src_sent2pair_nodes.append(s)
                dst_sent2pair_nodes.append(i)
    
    gps_subgraph_data = {
        ('pair', 'ps', 'sent'): (src_pair2sent_nodes, dst_pair2sent_nodes),
        ('sent', 'sp', 'pair'): (src_sent2pair_nodes, dst_sent2pair_nodes)
    }
    gps_subgraph = dgl.heterograph(gps_subgraph_data)

    # add missing nodes in the bipartite graph
    distinct_sent_used = sorted(set(sent_used))
    if all_sentences[-1] > distinct_sent_used[-1]:
        gps_subgraph = dgl.add_nodes(gps_subgraph, all_sentences[-1] - distinct_sent_used[-1], ntype='sent')
    
    return gss_subgraph, gps_subgraph, gpp_subgraph

def read_docred(file_in, tokenizer, max_seq_length=1024, train=True):
    i_line = 0
    pos_samples = 0
    neg_samples = 0
    features = []
    if file_in == "":
        return None
    with open(file_in, "r") as fh:
        data = json.load(fh)

    for _, sample in enumerate(tqdm(data, desc="Example")):

        sents = []
        sent_map = []

        entities = sample['vertexSet']
        entity_start, entity_end = [], []
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start.append((sent_id, pos[0],))
                entity_end.append((sent_id, pos[1] - 1,))

        for i_s, sent in enumerate(sample['sents']):
            new_map = {}
            for i_t, token in enumerate(sent):
                tokens_wordpiece = tokenizer.tokenize(token)
                if (i_s, i_t) in entity_start:
                    tokens_wordpiece = ["*"] + tokens_wordpiece
                if (i_s, i_t) in entity_end:
                    tokens_wordpiece = tokens_wordpiece + ["*"]
                new_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)
            new_map[i_t + 1] = len(sents)
            sent_map.append(new_map)

        train_triple = {}

        if "labels" in sample and train:
            if len(sample['labels']) == 0:   # docs have no labels
                continue

            for label in sample['labels']:

                h, t = entities[label['h']], entities[label['t']]
                h_sent = [s['sent_id'] for s in h]
                t_sent = [s['sent_id'] for s in t]
                raw_evidence = h_sent + t_sent
                raw_evidence = list(set(raw_evidence))

                evidence = label['evidence']
                r = int(docred_rel2id[label['r']])
                if (label['h'], label['t']) not in train_triple:
                    train_triple[(label['h'], label['t'])] = [
                        {'relation': r, 'evidence': evidence, 'raw_evidence': raw_evidence}]
                else:
                    train_triple[(label['h'], label['t'])].append(
                        {'relation': r, 'evidence': evidence, 'raw_evidence': raw_evidence})

        entity_pos = []
        entity_pos_sentence = {}    # key: entity, values: sentences
        pos = 0
        for e in entities:
            entity_pos.append([])
            sent_id = []
            for m in e:
                start = sent_map[m["sent_id"]][m["pos"][0]]
                end = sent_map[m["sent_id"]][m["pos"][1]]
                entity_pos[-1].append((start, end,))
                sent_id.append(m['sent_id'])
            sent_id = list(set(sent_id))
            entity_pos_sentence[pos] = sent_id
            pos += 1

        relations, hts, raw_evidences = [], [], []
        pos_hts = []
        pair_evidences, triple_evidences = [], []
        for h, t in train_triple.keys():
            relation = [0] * len(docred_rel2id)
            raw_evidence, pair_evidence = [], []
            for mention in train_triple[h, t]:
                relation[mention["relation"]] = 1
                evidence = mention["evidence"]
                raw_evidence.extend(mention['raw_evidence'])
                pair_evidence.extend(evidence)
                triple_evidences.append({
                    "h": h,
                    "t": t,
                    "r": mention['relation'],
                    'evidence': mention['evidence']
                })
            raw_evidence = list(set(raw_evidence))
            pair_evidence = list(set(pair_evidence))
            relations.append(relation)
            raw_evidences.append(raw_evidence)
            pair_evidences.append(pair_evidence)
            hts.append([h, t])
            pos_hts.append([h, t])
            pos_samples += 1

        for h in range(len(entities)):
            for t in range(len(entities)):
                if h != t and [h, t] not in hts:
                    relation = [1] + [0] * (len(docred_rel2id) - 1)
                    relations.append(relation)
                    
                    if not train:
                        h_e = entities[h]
                        t_e = entities[t]
                        raw_evidence = set([])
                        for h_e_mention in h_e:
                            raw_evidence.add(h_e_mention['sent_id'])
                        for t_e_mention in t_e:
                            raw_evidence.add(t_e_mention['sent_id'])
                        raw_evidences.append(list(raw_evidence))
                    
                    hts.append([h, t])
                    neg_samples += 1

        assert len(relations) == len(entities) * (len(entities) - 1)

        sents = sents[:max_seq_length - 2]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        i_line += 1
        new_sent_map = []
        for s in sent_map:
            s_pos = list(s.values())
            new_sent_map.append([s_pos[0], s_pos[-1]])

        # Building graph
        inputs = {
            'hts': pos_hts if train else hts,
            'raw_evidences': raw_evidences,
            'doc_length': len(new_sent_map),
            'entity_pos_sentence': entity_pos_sentence
        }
        gss_subgraph, gps_subgraph, gpp_subgraph = build_graphs(**inputs)

        feature = {
            'input_ids': input_ids,
            'entity_pos': entity_pos,
            'labels': relations,
            'hts': hts,
            'title': sample['title'],
            'sent_map': new_sent_map,
            'raw_evidence': raw_evidences,
            'gss_subgraph': gss_subgraph,
            'gps_subgraph': gps_subgraph,
            'pair_evidence': pair_evidences,
            'triple_evidence': triple_evidences,
            'gpp_subgraph': gpp_subgraph
        }
        
        features.append(feature)

    print("# of documents {}.".format(i_line))
    print("# of positive examples {}.".format(pos_samples))
    print("# of negative examples {}.".format(neg_samples))
    return features