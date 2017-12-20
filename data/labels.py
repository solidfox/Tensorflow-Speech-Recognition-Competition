__author__ = 'Daniel Schlaug'


_all_label_strings = 'yes no up down left right on off stop go silence unknown'.split()
_accepted_as_unknown = set(filter(lambda label: label not in _all_label_strings, "bed bird cat dog down eight five four go happy house left marvin nine no off on one right seven sheila six stop three tree two up wow yes zero".split()))
_set = set(_all_label_strings)
_name2index = {name: i for i, name in enumerate(_all_label_strings)}


class Label:

    def __init__(self, string):
        if string in _set:
            self.string = string
            self.index = _name2index[string]
        else:
            raise AttributeError(string + " is not a valid Label.")

    def __repr__(self):
        return "Label.{}".format(self.string)


class Label(Label):
    yes = Label("yes")
    no = Label("no")
    up = Label("up")
    down = Label("down")
    left = Label("left")
    right = Label("right")
    on = Label("on")
    off = Label("off")
    stop = Label("stop")
    go = Label("go")
    silence = Label("silence")
    unknown = Label("unknown")
    all_labels = [yes, no, up, down, left, right, on, off, stop, go, silence, unknown]

    @staticmethod
    def from_string(string):
        labels = filter(lambda label: label.string == string, Label.all_labels)
        if len(labels) == 1:
            return labels[0]
        elif string == "_background_noise_":
            return Label.silence
        elif string in Label._accepted_as_unknown:
            return Label.unknown
        else:
            raise AttributeError(string + " is not a valid label-string.")

    @staticmethod
    def from_index(index):
        return Label.all_labels[index]
