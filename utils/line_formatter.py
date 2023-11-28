from typing import Dict, Any


class LineFormatter:
    def __init__(self,
                 column_width: int = 8,
                 seperator: str = " | ",
                 extra_indent: int = 18):
        self.column_width = column_width
        self.seperator = seperator
        self.extra_indent = extra_indent  # To compensate for date in logging

        self.metric_names = None
        self.print_header_in = 0
        self.header = None

    def create_header(self):
        cw = self.column_width
        print_names = [mn.replace("_", " ").title() + ":"
                       for mn in self.metric_names]
        print_names = [mn.replace("Train ", "Tr") for mn in print_names]
        print_names = [mn.replace("Val ", "Vl") for mn in print_names]
        print_names = [f"{name:{cw}.{cw}}" for name in print_names]
        header = self.seperator.join(print_names)
        return header

    def create_line(self, logs: Dict[str, Any]):
        cw = self.column_width

        if self.header is None:
            self.metric_names = sorted(list(logs.keys()))
            self.header = self.create_header()

        line = ""

        if self.print_header_in == 0:
            self.print_header_in = 10
            line = self.header + "\n" + " "*self.extra_indent + line

        self.print_header_in -= 1

        value_strs = []
        for metric_name in self.metric_names:
            value = logs.get(metric_name, -1.)
            if isinstance(value, float):
                value_str = f"{value:5.4g}"
            else:
                value_str = str(value)
            value_strs.append(f"{value_str:{cw}.{cw}}")

        line += self.seperator.join(value_strs)
        return line
