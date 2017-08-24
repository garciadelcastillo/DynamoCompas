from __future__ import print_function

from compas.cad.rhino.forms import Form

try:
    from System.Windows.Forms import TextBox
    from System.Windows.Forms import DockStyle
    from System.Windows.Forms import ScrollBars
    from System.Drawing import Font
    from System.Drawing import FontFamily
    from System.Environment import NewLine

except ImportError:
    import platform
    if platform.python_implementation() == 'IronPython':
        raise


__author__     = ['Tom Van Mele', ]
__copyright__  = 'Copyright 2014, BLOCK Research Group - ETH Zurich'
__license__    = 'MIT License'
__email__      = 'vanmelet@ethz.ch'


__all__ = ['TextForm', ]


class TextForm(Form):
    """"""

    def __init__(self, text, title='TextForm', width=800, height=600):
        self.text = text
        super(TextForm, self).__init__(title, width, height)

    def init(self):
        textbox = TextBox()
        textbox.ReadOnly = True
        textbox.Dock = DockStyle.Fill
        textbox.Multiline = True
        textbox.ScrollBars = ScrollBars.Vertical
        textbox.Font = Font(FontFamily.GenericMonospace, 8.0)
        if isinstance(self.text, (str, unicode)):
            textbox.Text = self.text
        else:
            try:
                textbox.Text = (NewLine).join(self.text)
            except Exception as e:
                textbox.Text = str(e)
        self.Controls.Add(textbox)


# ==============================================================================
# Debugging
# ==============================================================================

if __name__ == '__main__':

    from this import d
    from this import s

    print(d)
    print(s)

    zen = ''.join([x if x not in d else d[x] for x in s])
    zen = zen.split('\n')

    form = TextForm(zen)
    form.show()
