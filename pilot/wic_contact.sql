-- wic_predictions summarizes the cornerstone wic data for a given kid_id
create table wic_contact as (
with highest_risk as (
    select distinct on (kid_id) kid_id, address_id
    from predictions order by kid_id, score desc
),

wic_address_ids as (
    select kid_id, address_id,
        ka.wic_infant_ogc_fids is not null as address_wic_infant,
        ka.wic_mother_ogc_fids is not null as address_wic_mother,
        p.address_test_min_date is not null as address_test,
        unnest(coalesce(ka.wic_infant_ogc_fids || ka.wic_mother_ogc_fids, 
                '{NULL}'::integer[])) ogc_fid,
        highest_risk.address_id is not null as address_primary
    from predictions p
    join output.kid_addresses ka using (kid_id, address_id)
    left join highest_risk using (kid_id, address_id)
)

select kid_id, address_id, address,
    array_remove(array_agg(distinct trim(addr_ln1_t)), null) addr_ln1_t,
    array_remove(array_agg(distinct trim(addr_ln2_t)), null) addr_ln2_t,
    array_remove(array_agg(distinct trim(addr_apt_t)), null) addr_apt_t,
    array_remove(array_agg(distinct addr_zip_n), null) addr_zip_n,
    array_remove(array_agg(distinct 
        CASE WHEN address_wic_infant THEN trim(
            -- change 'DOE, JANE' to 'JANE DOE'
            CASE WHEN position(',' in cont_nme_t) > 0 THEN
                substring(cont_nme_t from position(',' in cont_nme_t) + 1) 
                || ' ' || split_part(cont_nme_t, ',', 1)
            ELSE cont_nme_t END
        )
        -- when it's the mother's info use her as the contact
        ELSE brth_fst_t || ' ' || brth_lst_t END
    ), null) cont_nme_t,
    array_remove(array_agg(distinct 
        CASE WHEN address_wic_infant 
             THEN relate_c ELSE 'MO' END), null) relate_c,
    array_remove(array_agg(distinct nullif(phne_nbr_n, 0)), null) phne_nbr_n,
    array_remove(array_agg(distinct nullif(cell_nbr_n, 0)), null) cell_nbr_n,
    bool_or(address_wic_infant) as address_wic_infant,
    bool_or(address_wic_mother) as address_wic_mother,
    bool_or(address_test) as address_test,
    bool_or(address_primary) as address_primary
from wic_address_ids
join aux.addresses using (address_id)
left join cornerstone.partaddr using (ogc_fid)
left join cornerstone.partenrl on addr_id_i = part_id_i
group by kid_id, address_id, address
);

