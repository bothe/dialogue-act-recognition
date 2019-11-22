import matplotlib.pyplot as plt
import numpy as np


def plot_eda_usage(labels, label_values, title, colors_emo,
                   sentiments, sentiments_values, colors_sent,
                   test_show_plot=False, data_name='meld', plot_pie=True):
    # Create a pieplot
    # label_values.append(sum(label_values))
    # labels.append('White')
    print(len(labels), labels)
    data = label_values[0:len(labels)-1] + [label_values[-1]]
    print(len(data), data)

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
            plt.bar(labels[i], data[i]) #, colors=colors_emo)


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

# create data
# emotions = ['12', '11', '3', '30', '0', '23', '4']
# values = [12, 11, 3, 30, 0, 23, 4]
# colors_emo = ['Green', 'Red', 'Orange', 'Blue', 'Magenta', 'Black', 'Gray', 'White']
# sentiments = ['12', '11', '3']
# sentiments_values = [12, 11, 3]
# colors_sent = ['Green', 'Red', 'Orange', 'Blue', 'Magenta', 'Black', 'Gray', 'White']
# title = 'tag\nHello World'
# plot_eda_usage(emotions, values, title, colors_emo,
#                sentiments, sentiments_values, colors_sent, test_show_plot=True, plot_pie=False)
# plot_normal_bars(emotions, values, title, test_show_plot=True)
