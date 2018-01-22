from PIL import Image
import sys
from pathlib import Path
import pyocr
import pyocr.builders

tools = pyocr.get_available_tools()
assert(len(tools) != 0)
tool = tools[0]
langs = tool.get_available_languages()
lang = langs[0]
print(tool.get_name(), lang)

txt = tool.image_to_string(
    Image.open(str(Path.home() / 'Desktop' / 'adversarial-example.png')),
    lang=lang,
    builder=pyocr.builders.TextBuilder()
)
