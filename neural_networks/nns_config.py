""" Configuration of neural network models. """


def list_to_string(list):
    """Helper function to transform lists of integers to strings, in a possibly concise way."""
    if len(list) == 1:
        string = '{}x1'.format(list[0])
    elif list[1:] == list[:-1]:
        string = '{}x{}'.format(list[1], len(list))
    else:
        string = ''
        for i in range(len(list) - 1):
            string += str(list[i]) + ','
        string += str(list[-1])
    return string


nns_config = dict()

################
# STACKED LSTM #
################
nns_config['stacked_lstm'] = {
    'rnn_type': 'lstm',  # 'lstm', 'gru'
    'lstm_units': [256] * 2,
    'lstm_dropout': .3,
    'residual': 'none',  # alternatives 'residual', 'highway', 'none'
    'high_init': -1,
    'dense_units': [512] * 2,
    'dense_dropout': .5,
    'lr': 4e-4,
    'model_name': 'stacked_'
}

nns_config['stacked_lstm']['model_name'] += '{}_{}_dense{}'.format(
    nns_config['stacked_lstm']['rnn_type'],
    list_to_string(nns_config['stacked_lstm']['lstm_units']),
    list_to_string(nns_config['stacked_lstm']['dense_units']))
if nns_config['stacked_lstm']['residual'] != 'none':
    nns_config['stacked_lstm']['model_name'] += '_' + nns_config['stacked_lstm']['residual']

##############
# TCN + LSTM #
##############
nns_config['tcn_lstm'] = {
    'rnn_type': 'lstm',  # 'lstm', 'gru'
    'channels': [400],
    'kernel_size': [3],
    'dropout': 0.1,
    'stride': [1],
    'use_highway': False,
    'high_init': -1,  # 0 for glorot uniform initialization, otherwise negative constant initialization (-1, -3)
    'use_glu': True,
    'attention': False,
    'att_dropout': .5,
    'keep_len': True,
    'causal': False,
    'use_mask': True,
    'lstm_dropout': .3,
    'lstm_units': 256,
    'dense_units': [512] * 2,
    'dense_dropout': .5,
    'lr': 4e-4,
    'model_name': 'tcn_'
}

nns_config['tcn_lstm']['model_name'] += '{}_{}_{}_lstm_{}_dense{}'.format(
    nns_config['tcn_lstm']['rnn_type'],
    list_to_string(nns_config['tcn_lstm']['channels']),
    list_to_string(nns_config['tcn_lstm']['kernel_size']),
    nns_config['tcn_lstm']['lstm_units'],
    list_to_string(nns_config['tcn_lstm']['dense_units']))
if nns_config['tcn_lstm']['use_glu']:
    nns_config['tcn_lstm']['model_name'] += '_glu'
if nns_config['tcn_lstm']['use_highway']:
    nns_config['tcn_lstm']['model_name'] += '_high{}'.format(nns_config['tcn_lstm']['high_init'])
if nns_config['tcn_lstm']['attention']:
    nns_config['tcn_lstm']['model_name'] += '_att'

##############
# DOUBLE TCN #
##############
nns_config['double_tcn'] = {
    'channels_1': [400],
    'kernel_size_1': [3],
    'dropout_1': 0.1,
    'stride_1': [1],
    'use_highway_1': False,
    'high_init_1': -1,  # 0 for glorot uniform initialization, otherwise negative constant initialization (-1, -3)
    'use_glu_1': True,
    'attention_1': False,
    'att_dropout_1': 0.5,
    'keep_len_1': True,
    'causal_1': False,
    'intermediate_dropout': 0.3,
    # second tcn
    'channels_2': [256, 128, 64],
    'kernel_size_2': [3] * 3,
    'dropout_2': 0.1,
    'stride_2': [1] * 3,
    'use_highway_2': False,
    'high_init_2': -1,  # 0 for glorot uniform initialization, otherwise negative constant initialization (-1, -3)
    'use_glu_2': True,
    'attention_2': False,
    'att_dropout_2': 0.5,
    'keep_len_2': False,
    'causal_2': False,
    'dense_units': [512] * 2,
    'dense_dropout': 0.5,
    'lr': 4e-4,
    'model_name': 'double_tcn_',
}

nns_config['double_tcn']['model_name'] += 'c1_{}_k1_{}_c2_{}_k2_{}_dense{}'.format(
    list_to_string(nns_config['double_tcn']['channels_1']),
    list_to_string(nns_config['double_tcn']['kernel_size_1']),
    list_to_string(nns_config['double_tcn']['channels_2']),
    list_to_string(nns_config['double_tcn']['kernel_size_2']),
    list_to_string(nns_config['double_tcn']['dense_units']))

if nns_config['double_tcn']['use_glu_1']:
    nns_config['double_tcn']['model_name'] += '_glu1'
if nns_config['double_tcn']['use_highway_1']:
    nns_config['double_tcn']['model_name'] += '_high1{}'.format(nns_config['double_tcn']['high_init_1'])
if nns_config['double_tcn']['attention_1']:
    nns_config['double_tcn']['model_name'] += '_att1'
if nns_config['double_tcn']['use_glu_2']:
    nns_config['double_tcn']['model_name'] += '_glu2'
if nns_config['double_tcn']['use_highway_2']:
    nns_config['double_tcn']['model_name'] += '_high2{}'.format(nns_config['double_tcn']['high_init_2'])
if nns_config['double_tcn']['attention_2']:
    nns_config['double_tcn']['model_name'] += '_att2'

#######
# CNN #
#######
nns_config['cnn'] = {
    'filters': [64] * 16,
    'kernel_sizes': [2, 3, 4, 6] * 4,
    'strides': [1, 1, 1, 1] * 4,
    'dilations': [1] * 12 + [2] * 4,
    'dense_dropout': 0.5,
    'dense_units': [512] * 2,
    'lr': 4e-4,
    'model_name': 'cnn_'
}

nns_config['cnn']['model_name'] += '{}_{}_{}_{}'.format(list_to_string(nns_config['cnn']['filters']),
                                                        list_to_string(nns_config['cnn']['kernel_sizes']),
                                                        list_to_string(nns_config['cnn']['dilations']),
                                                        list_to_string(nns_config['cnn']['dense_units']))

#############
# TCN + CNN #
#############
nns_config['tcn_cnn'] = {
    'channels': [400],
    'kernel_size': [3],
    'dropout': 0.1,
    'stride': [1],
    'use_highway': False,
    'high_init': -1,  # 0 for glorot uniform initialization, otherwise negative constant initialization (-1, -3)
    'use_glu': True,
    'attention': False,
    'att_dropout': 0.5,
    'keep_len': True,
    'causal': False,
    'pre_cnn_dropout': 0.3,
    # cnn
    'filters': [64] * 16,
    'kernel_sizes': [2, 3, 4, 6] * 4,
    'strides': [1, 1, 1, 1] * 4,
    'dilations': [1] * 12 + [2] * 4,
    'dense_dropout': 0.5,
    'dense_units': [512] * 2,
    'lr': 4e-4,
    'model_name': 'tcn_cnn_'
}

nns_config['tcn_cnn']['model_name'] += '{}_{}_cnn_{}_{}_{}_dense{}'.format(
    list_to_string(nns_config['tcn_cnn']['channels']),
    list_to_string(nns_config['tcn_cnn']['kernel_size']),
    list_to_string(nns_config['tcn_cnn']['filters']),
    list_to_string(nns_config['tcn_cnn']['kernel_sizes']),
    list_to_string(nns_config['tcn_cnn']['dilations']),
    list_to_string(nns_config['tcn_cnn']['dense_units']))
if nns_config['tcn_cnn']['use_glu']:
    nns_config['tcn_cnn']['model_name'] += '_glu'
if nns_config['tcn_cnn']['use_highway']:
    nns_config['tcn_cnn']['model_name'] += '_high{}'.format(nns_config['tcn_cnn']['high_init'])
if nns_config['tcn_cnn']['attention']:
    nns_config['tcn_cnn']['model_name'] += '_att'

#################
# RNN ATTENTION #
#################
nns_config['rnn_attention'] = {
    'rnn_type': 'lstm',
    'rnn_units': 1024,  # 512,
    'rnn_highway_mode': 'none',  # ['residual', 'highway', 'none']
    'rnn_high_init': -1,
    'rnn_use_weight_norm': False,
    'use_glu': False,
    'attention_mode': 'attention',  # ['none', attention, multihead_attention, 'transformer']

    'multihead_nheads' : 16, #4
    'multihead_dim_exp' : 8, #2
    
    'transformer_nlayers' : 2,
    'transformer_attn_heads' : 16,
    'transformer_attn_dim_exp' : 4,
    'transformer_feedfw_dim_exp' : 8,

    'dense_units': [512, 256],

    'rnn_in_dropout': 0.3,
    'rnn_out_dropout': 0.3,
    'dense_dropout': 0.5,

    'lr': 4e-4,
    'model_name' : 'rnn_attention_'
}

nns_config['rnn_attention']['model_name'] += '{}{}'.format(nns_config['rnn_attention']['rnn_type'], nns_config['rnn_attention']['rnn_units'])

if nns_config['rnn_attention']['rnn_highway_mode'] == 'residual':
    nns_config['rnn_attention']['model_name'] += '_residual'
elif nns_config['rnn_attention']['rnn_highway_mode'] == 'highway':
    nns_config['rnn_attention']['model_name'] += '_highway_{}'.format(nns_config['rnn_attention']['rnn_high_init'])

if nns_config['rnn_attention']['rnn_use_weight_norm'] :
    nns_config['rnn_attention']['model_name'] += '_weightnorm'
    
if nns_config['rnn_attention']['use_glu'] :
    nns_config['rnn_attention']['model_name'] += '_glu'

if nns_config['rnn_attention']['attention_mode'] != 'none':
    nns_config['rnn_attention']['model_name'] += '_{}'.format(nns_config['rnn_attention']['attention_mode'])

nns_config['rnn_attention']['model_name'] += '_dense'
for u in nns_config['rnn_attention']['dense_units']:
    nns_config['rnn_attention']['model_name'] += 'x{}'.format(u)

###############
# TRANSFORMER #
###############
nns_config['transformer'] = {
    'transformer_layers': 2,  # 6, 12
    'transformer_attn_heads': 16,  # 8, 12
    'transformer_attn_dim_exp': 4,  # 4
    'transformer_feedfw_dim_exp': 8,  # 8, 4
    'transformer_attn_use_glu_head': False,
    'transformer_attn_use_glu_out': False,
    'transformer_feedfw_use_glu': False,
    'transformer_drop_hl_in': .1,
    'transformer_drop_attn_intern': .1,
    'transformer_drop_attn_exit': .3,
    'transformer_drop_feedfw': .3,
    'transformer_drop_hl_out': .2,
    'transformer_return_sequences': False,
    'transformer_add_pos_encodings': True,

    'dense_units': [512, 512],
    'lr': 4e-4,   #3e-4
    'transformer_dropout': 0.4,   #.3
    'dense_dropout': 0.5,   #.3
    
    'model_name' : 'transformer_'
}

nns_config['transformer']['model_name'] += '{}l_{}h_{}atn_{}fw'.format(
    nns_config['transformer']['transformer_layers'],
    nns_config['transformer']['transformer_attn_heads'],
    nns_config['transformer']['transformer_attn_dim_exp'],
    nns_config['transformer']['transformer_feedfw_dim_exp'])

if nns_config['transformer']['transformer_attn_use_glu_head']:
    nns_config['transformer']['model_name'] += '_gluhead'
if nns_config['transformer']['transformer_attn_use_glu_out']:
    nns_config['transformer']['model_name'] += '_gluout'
if nns_config['transformer']['transformer_feedfw_use_glu']:
    nns_config['transformer']['model_name'] += '_glufw'
if nns_config['transformer']['transformer_return_sequences']:
    nns_config['transformer']['model_name'] += '_retseq'
if nns_config['transformer']['transformer_add_pos_encodings']:
    nns_config['transformer']['model_name'] += '_addpos'

nns_config['transformer']['model_name'] += '_dense'
for u in nns_config['transformer']['dense_units']:
    nns_config['transformer']['model_name'] += 'x{}'.format(u)


###########
# MLP :-) #
###########

nns_config['mlp'] = {
    'model_name': 'mlp',
    'dense_units': [256, 256],
    'dense_dropout': 0.3
}
