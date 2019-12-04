import holoviews as hv
import pandas as pd
from bokeh.plotting import show
from bokeh.sampledata.les_mis import data
from holoviews import opts, dim
import pandas as pd


def plot_sankey(label, color, source, target, value, color_links):
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Sankey(
        node = dict(pad = 25, thickness = 20, line = dict(color = "green", width = 0.), label = label, color = color),
        link = dict(source = source, target = target, value = value, color = color_links))])
    fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)
    fig.show()


label = ["A1", "A2", "A3", "B1", "B2", "B3"]
color = ["rgba(242, 116, 32, 1.)", "rgba(0, 116, 32, 1.)", "rgba(242, 0, 32, 1.)",
         "antiquewhite", "aqua", "aquamarine"]
source = [0, 0, 0, 1, 1, 1, 2, 2, 2]
target = [3, 4, 5, 3, 4, 5, 3, 4, 5]
value = [8, 8, 12, 8, 8, 6, 8, 8, 6]
color_links = ["rgba(242, 116, 32, 1.)", "rgba(242, 116, 32, 0.5)", "rgba(242, 116, 32, 0.5)",
               "rgba(0, 116, 32, 1.)","rgba(0, 116, 32, 0.5)", "rgba(0, 116, 32, 0.5)",
               "rgba(242, 0, 32, 1.)", "rgba(242, 0, 32, 0.5)", "rgba(242, 0, 32, 0.5)"]
plot_sankey(label, color, source, target, value, color_links)

# Result is in "customer-good.png"
hv.extension('bokeh')
hv.output(size=200)

links = pd.DataFrame(data['links'])
print(links.head(3))

nodes = hv.Dataset(pd.DataFrame(data['nodes']), 'index')
nodes.data.head()

chord = hv.Chord((links, nodes)).select(value=(5, None))
chord.opts(opts.Chord(cmap='Category20', edge_cmap='Category20', edge_color=dim('source').str(),
           labels='name', node_color=dim('index').str()))
show(hv.render(chord))
