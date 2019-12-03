import pandas as pd
import holoviews as hv
from holoviews import opts, dim
from bokeh.sampledata.les_mis import data
from bokeh.plotting import show
from pysankey import sankey

hv.extension('bokeh')
hv.output(size=200)

links = pd.DataFrame(data['links'])
print(links.head(3))

nodes = hv.Dataset(pd.DataFrame(data['nodes']), 'index')
nodes.data.head()

chord = hv.Chord((links, nodes)).select(value=(5, None))
chord.opts(
    opts.Chord(cmap='Category20', edge_cmap='Category20', edge_color=dim('source').str(),
               labels='name', node_color=dim('index').str()))
show(hv.render(chord))
