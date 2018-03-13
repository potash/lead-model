from drain.step import Step

class LeadScore(Step):
    def run(self, scores, aux, y=None, test=None):
        """
        Args:
            scores: the address scores
            aux: dataframe including an address column
            y: optional outcomes, aligned with aux
        """
        if test is not None:
            aux = aux[test]
            if y is not None:
                y = y[test]

        merged = aux.merge(scores.reset_index()[['address', 'score']], on='address', how='left')
        merged.index = aux.index
        if y is not None:
            merged['true'] = y

        return {'y':merged}
