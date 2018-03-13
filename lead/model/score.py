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

        scores = scores.reset_index()
        merged = aux.merge(scores[['address', 'score']], on='address', how='left')
        merged.index = aux.index

        missing = merged.score.isnull()
        # if scores are null that means it's a "new" address
        # so we use the geography score
        # which is based on community area, ward, and census block
        if missing.sum() > 0:
            geography_columns = ['community_area_id', 'ward_id', 'census_block_id']
            geography_scores = scores[scores.address.isnull()]
            missing_scores = aux[missing].merge(geography_scores, on=geography_columns, how='left').score.values
            merged.loc[missing, 'score'] = missing_scores

        if y is not None:
            merged['true'] = y

        return {'y':merged}
