import yaml


class Tagged:
    def __init__(self, tag, value, style=None) -> None:
        self.tag = tag
        self.value = value
        self.style = style


class TaggedScalar(Tagged):
    def __init__(self, tag, value, style=None) -> None:
        super().__init__(tag, value, style)


class TaggedSequence(Tagged):
    def __init__(self, tag, value, style=None) -> None:
        super().__init__(tag, value, style)


class TaggedMapping(Tagged):
    def __init__(self, tag, value, style=None) -> None:
        super().__init__(tag, value, style)
