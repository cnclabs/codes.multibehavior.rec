
from load_data import load_df
import argparse
import datetime
import pandas as pd
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: sample/Trivago/Kkbox/Tmall/Rrocket/Rees46')
opt = parser.parse_args()
print(opt)

print("-- Starting @ %ss" % datetime.datetime.now())

sess_, index_, item_, act_, index_gap, train_dataset, test_dataset = load_df(opt.dataset)

# make sure no duplicate session_id (to split data later)
train_dataset[sess_] += '_tra'
test_dataset[sess_] += '_tes'

df = pd.concat([train_dataset, test_dataset], sort=True) 
df = df.dropna(subset = [sess_, index_, item_, act_]).reset_index(drop=True)

# Calculate item occurance
item_counts, act_counts = {}, {}
for item in df[item_]:
    if item in item_counts:
            item_counts[item] += 1
    else:
        item_counts[item] = 1

for act in df[act_]:
    if act in act_counts:
            act_counts[act] += 1
    else:
        act_counts[act] = 1

# convert to session sequence
df_seq = df[[sess_, index_, item_, act_]].sort_values([index_]).groupby(sess_).agg(list)
item_seq, action_seq = df_seq[item_].to_dict(), df_seq[act_].to_dict() # session_id, id_seq
print(df_seq.head())
print('session number:', len(item_seq))

# Filter data
for s in list(item_seq):
    # delete item/action occurance < 4 times
    # conduct on both seq to make sure they have same len
    pass_it, pass_act = [], [] # temp
    for i in range(len(item_seq[s])):
        if item_counts[item_seq[s][i]] > 3 and act_counts[action_seq[s][i]] > 3:
            pass_it += [item_seq[s][i]]
            pass_act += [action_seq[s][i]]
        
    item_seq[s], action_seq[s] = pass_it, pass_act

    # delete length 1 sessions (either item or action meets the condition, delete both)
    if len(item_seq[s]) < 2 or len(action_seq[s]) < 2:
        del item_seq[s]
        del action_seq[s]

# Split training and testing data
item_tra_seq = dict(filter(lambda x: '_tra' in x[0], item_seq.items()))
act_tra_seq = dict(filter(lambda x: '_tra' in x[0], action_seq.items()))
item_tes_seq = dict(filter(lambda x: '_tes' in x[0], item_seq.items()))
act_tes_seq = dict(filter(lambda x: '_tes' in x[0], action_seq.items()))

# Encode item & action id
def encode_seqs(seq, test=False, id_dict=None):
    code_dict = {}
    code = 1
    
    # renumber ids to start from 1
    for s in list(seq):
        encode_seq = []
        for i in seq[s]:
            if test:
                if i in id_dict:
                    encode_seq += [id_dict[i]]
                else:
                    encode_seq += [-1] # temp
            else:
                if i in code_dict:
                    encode_seq += [code_dict[i]]
                else:
                    encode_seq += [code]
                    code_dict[i] = code
                    code += 1
        seq[s] = encode_seq

    if test:
        return seq  
    return seq, code_dict, code-1

# Train
itra_seq, item_dict, item_n = encode_seqs(item_tra_seq)
atra_seq, act_dict, act_n = encode_seqs(act_tra_seq)

# Test
ites_seq = encode_seqs(item_tes_seq, test=True, id_dict=item_dict)
ates_seq = encode_seqs(act_tes_seq, test=True, id_dict=act_dict)

print('Total item:', item_n, 'Total action:', act_n)
print('Traing session:', len(itra_seq), 'Testing session:', len(ites_seq))

# align item & action seq
def align_seq(iseq, aseq):
    for s in list(iseq):
        pass_it, pass_act = [], [] # temp
        for i in range(len(iseq[s])):
            if iseq[s][i] != -1 and aseq[s][i] != -1:
                pass_it += [iseq[s][i]]
                pass_act += [aseq[s][i]]
        iseq[s], aseq[s] = pass_it, pass_act
        
        if len(iseq[s]) < 2 or len(aseq[s]) < 2:
            del iseq[s]
            del aseq[s]
    return iseq, aseq

itra_seq, atra_seq = align_seq(itra_seq, atra_seq)
ites_seq, ates_seq = align_seq(ites_seq, ates_seq)

def process_seq(seq, data=opt.dataset, index_gap=20):
    sess_id, out_seqs, labels = [], [], []

    for sid, i in seq.items():
        t = 0
        # divide different sessions
        if index_gap != None and len(i) > index_gap:
            for j in range(0, len(i), index_gap):
                now_s = i[j:j+index_gap]
                if len(now_s) > 2:
                    labels += [now_s[-1]]
                    out_seqs += [now_s[:-1]]
                    sess_id += [sid+'_'+str(t)]
                    t += 1
        else:
            labels += [i[-1]]
            out_seqs += [i[:-1]]
            sess_id += [sid]

    return sess_id, out_seqs, labels

# with data augmentation
def process_seq_aug(seq, data=opt.dataset):
    sess_id, out_seqs, labels = [], [], []
    for sid, i in seq.items():
        for j in range(1, len(i)):
            labels += [i[-j]]
            out_seqs += [i[:-j]]
            sess_id += [sid]
    return sess_id, out_seqs, labels

sess_id_tra, itra, itra_lab = process_seq(itra_seq, index_gap=index_gap)
_, atra, atra_lab = process_seq(atra_seq, index_gap=index_gap)

sess_id_tes, ites, ites_lab = process_seq(ites_seq, index_gap=index_gap)
_, ates, ates_lab = process_seq(ates_seq, index_gap=index_gap)

all = 0
for i in itra:
    all += len(i)
print('\nAverage training session length:', all/len(itra))

train = (itra, itra_lab, atra, atra_lab)
test = (ites, ites_lab, ates, ates_lab)

print('Training data:', len(train[0]), 'Testing data:', len(test[0]))
print('item\n', sess_id_tra[:3], itra[:3], itra_lab[:3])
print('action\n', atra[:3], atra_lab[:3])
print('item\n', sess_id_tes[:3], ites[:3], ites_lab[:3])
print('action\n', ates[:3], ates_lab[:3])

# Save data
if not os.path.exists(opt.dataset):
    os.makedirs(opt.dataset)
pickle.dump(train, open(opt.dataset+'/train.txt', 'wb'))
pickle.dump(test, open(opt.dataset+'/test.txt', 'wb'))
pickle.dump([item_n, act_n], open(opt.dataset+'/info.pkl', 'wb'))

print('Done!')