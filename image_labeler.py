import ipywidgets as widgets
from IPython.display import display, clear_output
from PIL import Image
from io import BytesIO
from typing import NamedTuple
import abc


class InputCellFactory(object):

    @abc.abstractmethod
    def new(self, result_callback, value=None):
        raise NotImplementedError()


class MultiClassInputCellFactory(InputCellFactory):
    """For labeling a fixed number of classes"""

    def __init__(self, classes, default=None):
        self._classes = classes
        self._default_class = default if default else classes[0]

    def new(self, result_callback, value=None):
        if value is None or value not in self._classes:
            value = self._default_class
        select_widget = widgets.ToggleButtons(
            options=self._classes,
            value=value
        )
        select_widget.style.button_width = 'auto'

        def on_change(change):
            result_callback(change['new'])
        select_widget.observe(on_change, names='value')
        return select_widget


class DetectorValidationInputCellFactory(InputCellFactory):
    """For validating detectors"""

    class ValResult(NamedTuple):
        tp: int
        fp: int
        fn: int

    def new(self, result_callback, value=None):
        if value is None:
            tp, fp, fn = 0, 0, 0
        else:
            tp, fp, fn = value.tp, value.fp, value.fn
        tp_widget = widgets.BoundedIntText(
            description='true (+)', value=tp, min=0, max=100,
            layout=widgets.Layout(width='auto')
        )
        fp_widget = widgets.BoundedIntText(
            description='false (+)', value=fp, min=0, max=100,
            layout=widgets.Layout(width='auto')
        )
        fn_widget = widgets.BoundedIntText(
            description='false (-)', value=fn, min=0, max=100,
            layout=widgets.Layout(width='auto')
        )
        submit_btn = widgets.Button(
            description='Submit',
            layout=widgets.Layout(width='auto'),
            button_style='danger'
        )

        def on_submit(unused):
            result_callback(DetectorValidationInputCellFactory.ValResult(
                tp=tp_widget.value,
                fp=fp_widget.value,
                fn=fn_widget.value
            ))
        submit_btn.on_click(on_submit)
        return widgets.HBox([tp_widget, fp_widget, fn_widget, submit_btn])


class ImageLabeler(object):

    def __init__(self, images, cell_factory, default_labels=None,
                 multi_mode=False):
        """
        images: a list of numpy arrays
        cell_factory: see above
        default_labels: initial labels (by default, each label is None)
        """
        self._images = images
        self._cell_factory = cell_factory

        if default_labels is None:
            default_labels = [None] * len(images)
        assert len(default_labels) == len(images)
        self._labels = default_labels
        self._seen = [False] * len(images)

        self._output = widgets.Output()
        self._img_output = widgets.Output()
        if multi_mode:
            self._render_multi()
        else:
            self._render_single()
        display(self._output)

    def remove(self):
        self._output.clear_output()

    def _load_image(self, idx):
        img_fp = BytesIO()
        img = Image.fromarray(self._images[idx])
        img.save(img_fp, format='png')
        return img_fp.getvalue()

    @property
    def all_seen(self):
        return all(self._seen)

    @property
    def labels(self):
        return self._labels

    def _render_multi(self):

        def render_img(idx):
            img_data = self._load_image(idx)

            def cell_callback(label):
                self._labels[idx] = label

            img_output = widgets.Output()
            with img_output:
                img_widget = widgets.Image(value=img_data, format='png')
                display(self._cell_factory.new(
                    cell_callback, value=self._labels[idx]
                ))
                display(img_widget)
                self._seen[idx] = True
            display(img_output)

        for i in range(len(self._images)):
            render_img(i)

    def _render_single(self):
        n = len(self._images)
        idx = 0

        label_widget = widgets.Label()
        pbar_widget = widgets.IntProgress(value=1, min=1, max=n)

        def set_progess():
            pbar_widget.value = idx + 1
            label_widget.value = '{} / {}'.format(idx + 1, n)
        set_progess()

        def next_img(unused):
            nonlocal idx
            if idx < n - 1:
                idx += 1
                render_img(idx)
            set_progess()

        def prev_img(unused):
            nonlocal idx
            if idx > 0:
                idx -= 1
                render_img(idx)
            set_progess()

        prev_btn = widgets.Button(
            description='Previous',
            layout=widgets.Layout(width='auto'),
        )
        prev_btn.on_click(prev_img)

        next_btn = widgets.Button(
            description='Next',
            layout=widgets.Layout(width='auto'),
        )
        next_btn.on_click(next_img)

        auto_next_box = widgets.Checkbox(
            value=True, description='auto advance', indent=False
        )

        img_output = widgets.Output()

        def render_img(idx):
            img_data = self._load_image(idx)

            def cell_callback(label):
                self._labels[idx] = label
                if idx < n - 1 and auto_next_box.value:
                    next_img(None)

            img_output.clear_output()
            with img_output:
                img_widget = widgets.Image(value=img_data, format='png')
                display(self._cell_factory.new(
                    cell_callback, value=self._labels[idx]
                ))
                display(img_widget)
                self._seen[idx] = True

        with self._output:
            display(widgets.HBox([
                prev_btn, pbar_widget, next_btn, label_widget, auto_next_box
            ]))
            display(img_output)
            render_img(idx)

    def __repr__(self):
        return self.__class__.__name__ + ': ' + repr(self.labels)
