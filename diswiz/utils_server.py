import numpy as np


def highDAClass(da, DAs):
    if da in DAs['Statement']:
        return 'Statement'
    elif da in DAs['Questions']:
        return 'Questions'
    elif da in DAs['Answers']:
        return 'Answers'
    elif da in DAs['Agreement']:
        return 'Agreement'
    elif da in DAs['BackwardFunction']:
        return 'Backward Function'
    elif da in DAs['ForwardFunction']:
        return 'Forward Function'
    elif da in DAs['CommInfoStatus']:
        return 'CommInfoStatus'
    elif da in DAs['Other']:
        return 'Other'
    elif da in DAs['NonSpeechVerbal']:
        return 'NonSpeechVerbal'


DA_names = ['Other Functions', 'Wh-Question', 'Declarative Yes-No-Question ',
            'Yes-No-Question', 'Statement-non-opinion', 'Action-Directive', 'Hedge',
            'Agree/Accept', 'Backchannel', 'Statement-opinion', 'Acknowledge Ans',
            'No Answer', 'Affirmative Answer', 'Backchannel-Que', 'Yes Answers', 'NonSpeech',
            'Uninterpretable', 'Appreciation', 'Reformulate', 'Repeat Phrase',
            'Rhetorical-Question', 'Other Answers', 'Self-talk', 'Open-Question',
            'Hold', 'Or-Clause', 'Open Option', 'Quotation', 'Completion',
            'Signal-non-understand', 'Maybe/Accept-part', 'Downplayer',
            'Tag-Question', 'Conv. Closing', 'Thanking', 'Reject', '3rd-party-talk ',
            'Negative Answer', 'Info-Request', 'Conv. Opening', 'Apology', 'Reject-part']


def returnDAname(da, tag, DA_names):
    return DA_names[tag.index(da)]


'''
Statement        = ['sd', 'sv']					                                #2
Questions        = ['qw', 'qy^d', 'qy', 'qh', 'qo', 'qrr', '^q', 'qw^d', '^g']	#9
Answers          = ['nn', 'na', 'ny', 'no']				                        #4
Agreement        = ['aa', 'ar', 'ad','^h', 'aap_am', 'arp_nd']		            #6
BackwardFunction = ['b', 'bk', 'bh', 'ba', 'bf', 'br', 'bd', '^2']		        #8
ForwardFunction  = ['fo_o_fw_"_by_bc', 'fc', 'ft', 'fp', 'fa', 'oo_co_cc']	    #7
CommInfoStatus   = ['%', 'b^m', 't1', 't3']				                        #4
Other            = ['h', '^q']					                                #2
NonSpeechVerbal  = ['x']					                            	    #1
'''

DAs = {
    'Statement': ['sd', 'sv'],
    'Questions': ['qw', 'qy^d', 'qy', 'qh', 'qo', 'qrr', '^q', 'qw^d', '^g'],
    'Answers': ['nn', 'na', 'ny', 'no', 'ng'],
    'Agreement': ['aa', 'ar', 'ad', '^h', 'aap_am', 'arp_nd'],
    'BackwardFunction': ['b', 'bk', 'bh', 'ba', 'bf', 'br', 'bd', '^2'],
    'ForwardFunction': ['fo_o_fw_"_by_bc', 'fc', 'ft', 'fp', 'fa', 'oo_co_cc', 'fo_o', 'fw'],
    'CommInfoStatus': ['%', 'b^m', 't1', 't3'],
    'Other': ['h', '^q'],
    'NonSpeechVerbal': ['x']
}


def prepare_output(predictions, tag, it_value):
    str_preds = []
    for item in sorted(predictions[0], reverse=1):
        str_preds.append(str(item))
    tags = []
    for item in list(np.argsort(predictions[0]))[::-1]:
        tags.append(tag[item])

    i = 0
    booln = np.array(sorted(predictions[0], reverse=1)) > 0.07
    das, confs, DAnames = [], [], []
    for i in range(len(booln)):
        if booln[i] == True:
            das.append(tags[i])
            DAnames.append(returnDAname(tags[i], tag, DA_names))
            confs.append(str(sorted(predictions[0], reverse=1)[i])[0:4])
        i += 1
    classes = highDAClass(tag[np.argmax(predictions)], DAs)
    #    print(confs[0][0:5])
    return it_value, classes, DAnames, confs
