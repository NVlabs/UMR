import dominate
from dominate.tags import *
import os


class HTML:
    def __init__(self, web_dir, title, image_subdir='', reflesh=0):
        """
        Initialize web page.

        Args:
            self: (todo): write your description
            web_dir: (str): write your description
            title: (str): write your description
            image_subdir: (str): write your description
            reflesh: (str): write your description
        """
        self.title = title
        self.web_dir = web_dir
        # self.img_dir = os.path.join(self.web_dir, )
        self.img_subdir = image_subdir
        self.img_dir = os.path.join(self.web_dir, image_subdir)
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        # print(self.img_dir)

        self.doc = dominate.document(title=title)
        if reflesh > 0:
            with self.doc.head:
                meta(http_equiv="reflesh", content=str(reflesh))

    def get_image_dir(self):
        """
        Get the directory where the image dir.

        Args:
            self: (todo): write your description
        """
        return self.img_dir

    def add_header(self, str):
        """
        Add a header to the header.

        Args:
            self: (todo): write your description
            str: (todo): write your description
        """
        with self.doc:
            h3(str)

    def add_table(self, border=1):
        """
        Add a table to the table.

        Args:
            self: (todo): write your description
            border: (todo): write your description
        """
        self.t = table(border=border, style="table-layout: fixed;")
        self.doc.add(self.t)

    def add_images(self, ims, txts, links, width=400):
        """
        Add images to the document.

        Args:
            self: (todo): write your description
            ims: (str): write your description
            txts: (todo): write your description
            links: (str): write your description
            width: (int): write your description
        """
        self.add_table()
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=os.path.join(link)):
                                img(style="width:%dpx" % width, src=os.path.join(im))
                            br()
                            p(txt)

    def save(self,file='index'):
        """
        Save the document to a file.

        Args:
            self: (todo): write your description
            file: (str): write your description
        """
        html_file = '%s/%s.html' % (self.web_dir,file)
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()


if __name__ == '__main__':
    html = HTML('web/', 'test_html')
    html.add_header('hello world')

    ims = []
    txts = []
    links = []
    for n in range(4):
        ims.append('image_%d.png' % n)
        txts.append('text_%d' % n)
        links.append('image_%d.png' % n)
    html.add_images(ims, txts, links)
    html.save()
