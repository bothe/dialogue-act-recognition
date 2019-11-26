import matplotlib.pyplot as plt
import numpy as np


def plot_eda_usage(labels, label_values, title, colors_emo,
                   sentiments, sentiments_values, colors_sent,
                   test_show_plot=False, data_name='meld', plot_pie=True):
    print(len(labels), labels)
    data = label_values[0:len(labels)-1] + [label_values[-1]]
    print(len(data), data)

    # Create a pieplot
    if plot_pie:
        n_emo = plt.pie(data, colors=colors_emo, counterclock=False, startangle=180, radius=1.2)
        for i in range(len(n_emo[0])):
            n_emo[0][i].set_alpha(0.8)

        if data_name == 'meld':
            data_sent = sentiments_values[0:3] + [sentiments_values[-1]]
            plt.pie(data_sent, colors=colors_sent, counterclock=False, startangle=180, radius=0.8)
        # add a circle at the center
        my_circle = plt.Circle((0, 0), 0.5, color='white')
        p = plt.gcf()
        p.gca().add_artist(my_circle)
        # plt.title(title, y=1)
        plt.text(0, -0.15, title, fontdict={'size': 15.0, 'horizontalalignment': 'center'})
    else:
        for i in range(len(labels)):
            plt.bar(labels[i]+labels[i+1], data[i]) #, colors=colors_emo)

    if test_show_plot:
        plt.show()
        return
    plt.savefig('figures/' + data_name + '/fig_' + title.split('\n')[0], bbox_inches='tight', transparent=True)
    plt.close()


def plot_normal_bars(labels, label_values, title, test_show_plot=False):
    plt.rcParams.update({'font.size': 16})
    plt.bar(labels, label_values)
    plt.title(title.split('\n')[0] + ' - ' + title.split('\n')[1])
    plt.xticks(rotation=15)
    # plt.yaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.ylim(.5, 5.5)
    # plt.xlim(.5, 5.5)
    # plt.xlabel('Emotions')
    # plt.ylabel('Number of Utterances')
    if test_show_plot:
        plt.show()
        return
    plt.savefig('figures/meld/fig_' + title.split('\n')[0])
    plt.close()


def plot_bars_plot(stack_emotions_values, emotions, colors_emo, tags,
                   test_show_plot=False, data='meld', type_of='emotion', save_eps=False):
    from scr.plot_bars import StackedBarGrapher
    stack_emo_names = {}
    das_stacked = np.array(stack_emotions_values).transpose()
    for i in range(len(emotions)):
        stack_emo_names[emotions[i]] = das_stacked[i]
    totals = das_stacked.sum(axis=0)
    stack_emo_bars = []
    for key in stack_emo_names.keys():
        stack_emo_bars.append([round(i / j * 100, 3) for i, j in zip(stack_emo_names[key], totals)])
    bars = np.array(stack_emo_bars[0:9]).transpose()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    SBG = StackedBarGrapher()
    SBG.stackedBarPlot(ax, bars, colors_emo, xLabels=tags, gap=1.5, widths=[6.]*len(tags))
    if test_show_plot:
        plt.show()
        return
    if save_eps:
        plt.savefig('figures/' + data + '_bars_' + type_of+ '.eps', format='eps', bbox_inches='tight', transparent=True)
    else:
        plt.savefig('figures/' + data + '_bars_' + type_of, bbox_inches='tight', transparent=True)
    plt.close()
