import holoviews as hv
import pandas as pd
from bokeh.plotting import show
from bokeh.sampledata.les_mis import data
from holoviews import opts, dim
import pandas as pd


import plotly.graph_objects as go

fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(color = "green", width = 0.),
      label = ["A1", "A2", "A3", "B1", "B2", "B3"],
      color = ["red", "green", "aliceblue", "antiquewhite", "aqua", "aquamarine"]
    ),
    link = dict(
      source = [0, 0, 0, 1, 1, 1, 2, 2, 2], # indices correspond to labels, eg A1, A2, A2, B1, ...
      target = [3, 4, 5, 3, 4, 5, 3, 4, 5],
      value = [8, 8, 12, 8, 8, 6, 8, 8, 6],
      color = ["#F27420", "#F27420", "#F27420", "green", "green", "green", "red", "red", "red"]
))])
fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)
fig.show()


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
