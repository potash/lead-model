from drain.step import Step

class LeadScore(Step):
    def run(self, scores, aux, y=None):
        """
        Args:
            scores: the address scores
            aux: dataframe including an address column
            y: optional outcomes, aligned with aux
        """
        merged = aux.merge(scores.reset_index()[['address', 'score']], on='address', how='left')
        merged.index = aux.index
        if y is None:
            merged['y'] = y

        return y
