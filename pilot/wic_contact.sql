-- wic_predictions summarizes the cornerstone wic data for a given kid_id
create table wic_contact as (
with wic_address_ids as (
    select kid_id, address_id,
        address_wic_infant,
        address_wic_mother,
        unnest(coalesce(ka.wic_infant_ogc_fids || ka.wic_mother_ogc_fids, 
                '{NULL}'::integer[])) ogc_fid
    from output.kid_addresses ka
    where address_wic_min_date is not null
)

select kid_id, address_id, partaddr.ogc_fid, partaddr.last_upd_d,
    trim(addr_ln1_t) addr_ln1_t,
    trim(addr_ln2_t) addr_ln2_t,
    trim(addr_apt_t) addr_apt_t,
    addr_zip_n addr_zip_n,
    CASE WHEN address_wic_infant THEN trim(
        -- change 'DOE, JANE' to 'JANE DOE'
        CASE WHEN position(',' in cont_nme_t) > 0 THEN
            substring(cont_nme_t from position(',' in cont_nme_t) + 1) 
            || ' ' || split_part(cont_nme_t, ',', 1)
        ELSE cont_nme_t END
    )
    -- when it's the mother's info use her as the contact
    ELSE brth_fst_t || ' ' || brth_lst_t END AS cont_nme_t,
    CASE WHEN address_wic_infant 
         THEN relate_c ELSE 'MO' END relate_c,
    nullif(phne_nbr_n, 0) phne_nbr_n,
    nullif(cell_nbr_n, 0) cell_nbr_n,
    address_wic_infant as address_wic_infant,
    address_wic_mother as address_wic_mother
from wic_address_ids
join aux.addresses using (address_id)
left join cornerstone.partaddr using (ogc_fid)
left join cornerstone.partenrl on addr_id_i = part_id_i
);

