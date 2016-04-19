select taxpayer_address,
    count(distinct kid_id),
    avg(max_bll),
    array_agg(distinct taxpayer_name),
    array_agg(a0.address)

    from output.kids join output.addresses a0 on first_bll6_address_id = address_id join output.addresses a using (complex_id) join aux.properties p on a.address = p.address where date_of_birth > '2010-01-01'

    and taxpayer_name != 'EXEMPT'
    group by taxpayer_address having count(distinct kid_id) > 2 order by count(distinct kid_id) desc;
