from drain.step import Step

class GetItem(Step):
    def __init__(self, step, key):
        Step.__init__(self, step=step, key=key, inputs=[step])

    def run(self, **kwargs):
        return kwargs[self.key]

def args_list(*args):
    return args
