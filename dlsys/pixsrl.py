import webbrowser
from tempfile import NamedTemporaryFile

import tensorflow as tf


def show_graph(graph=None):
    write_visualization_html(graph or tf.get_default_graph())


def write_visualization_html(graph, html_file_path=None, max_const_size=32, show_in_browser=True):
    """
    Visualize TensorFlow graph as a static TensorBoard page.
    Notice that in order to view it directly, the user must have
    a working Chrome browser.
    The page directly embed GraphDef prototxt so that the page (and an active Internet connection)
    is only needed to view the content. There is NO need to fire up a web server in the backend.
    :param html_file_path: str, path to the HTML output file
    :param max_const_size: int, if a constant is way too long, clip it in the plot
    :param show_in_browser: bool, indicate if we want to launch a browser to show the generated HTML page.
    """
    _tfb_url_prefix = "https://tensorboard.appspot.com"
    _tfb_url = "{}/tf-graph-basic.build.html".format(_tfb_url_prefix)

    def strip_consts(gdef, max_const_size=32):
        """Strip large constant values from graph_def."""
        strip_def = tf.GraphDef()
        for n0 in gdef.node:
            n = strip_def.node.add()  # pylint: disable=E1101
            n.MergeFrom(n0)
            if n.op == 'Const':
                tensor = n.attr['value'].tensor
                nbytes = len(tensor.tensor_content)
                if nbytes > max_const_size:
                    tensor.tensor_content = str.encode("<stripped {} bytes>".format(nbytes))
        return strip_def

    strip_def = strip_consts(graph.as_graph_def(), max_const_size)
    html_code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="{tfb}" onload=load()>
        <div>
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)),
               id='gfn-sess-graph',
               tfb=_tfb_url)

    if not html_file_path:
        html_file_path = NamedTemporaryFile(prefix="tf_graph_def", suffix=".html").name

    # Construct the graph def board and open it
    with open(str(html_file_path), 'wb') as fout:
        try:
            fout.write(html_code)
        except TypeError:
            fout.write(html_code.encode('utf8'))

    if show_in_browser:
        webbrowser.get("chrome").open("file://{}".format(str(html_file_path)))
