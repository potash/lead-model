from drain import Step

class LeadData(Step)
    def __init__(self, month, day, year_min=2005, year_max=2015, **kwargs):
        Step.__init__(self, month=month, day=day, year_min=year_min, year_max=year_max)

    def run(self):
        engine = util.create_engine()
