-- wic_predictions summarizes the cornerstone wic data for a given kid_id
create temp table wic_contact as (
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

do $$
    DECLARE RISK_COUNT int := 300;
    DECLARE BASE_COUNT int := 0;
    DECLARE RANDOM_SEED double precision := 0;
    DECLARE DOB_MIN date := '2015-11-01'::date;
    DECLARE DOB_MAX date := '2016-09-30'::date;
BEGIN

-- subset of predictions features for kids with eligible dob
create temp table wic_kids as (
    with kids as (
        select distinct on (kid_id) kid_id, score, 
            address_id, address, first_name, last_name, date_of_birth, max_bll, test_count, last_sample_date
        from predictions
        where first_wic_date is not null 
            and date_of_birth 
            between DOB_MIN and DOB_MAX
        order by kid_id, score desc),
    -- get participant ids for kid
    part_id_is as (
        select kid_id, array_agg(distinct part_id_i) as part_id_i
        from predictions
        join aux.kid_wics using (kid_id)
        group by kid_id
    )
    select kids.*, new.part_id_i from kids 
    join part_id_is new using (kid_id)
    where 
    not (part_id_i && (select anyarray_agg(part_id_i) from pilot.pilot01))
    and not (address in (select address from pilot.pilot01_contact))
);
-- select risk group
create temp table pilot01_risk as (
    select *, false as inspection
    -- order by kid_id too in case there are score ties
    from wic_kids order by score desc, kid_id asc
    limit RISK_COUNT
);

-- set random half of addresses to receive inspection
PERFORM setseed(RANDOM_SEED);
update pilot01_risk set inspection = true 
where address_id in (
    select address_id from pilot01_risk 
    order by random() limit RISK_COUNT/2
);

-- select 20 for base group
PERFORM setseed(RANDOM_SEED);
create temp table pilot01_base as (
    select *, false as inspection from wic_kids
    where kid_id not in (select kid_id from pilot01_risk)
    order by random() limit BASE_COUNT
);

-- set random half to receive inspection
PERFORM setseed(RANDOM_SEED);
update pilot01_base set inspection = true 
where address_id in (
    select address_id from pilot01_base 
    order by random() limit BASE_COUNT/2
);

end $$;

drop table if exists pilot01;
create table pilot01 as
((select *, true as risk from pilot01_risk) UNION ALL
    (select *, false as risk from pilot01_base));
-- copy concatenated to csv

\copy pilot01 to data/pilot/02.csv with csv header;

drop table if exists pilot01_contact;
create table pilot01_contact as (
    with last_investigations as (
        select distinct on(address_id) *
        from output.investigations
        order by address_id, coalesce(closure_date, comply_date,init_date,referral_date) desc
    )
    select part_id_i, first_name, pilot01.last_name, date_of_birth, inspection, wic.*,
        phne_nbr_n[1] as phne_nbr_n1,   -- first phone number, which is usually the right one
        kid_ethnicity as ethnicity,      -- ethnicity, for Spanish language contact
        referral_date, init_date, hazard_int, hazard_ext, comply_date, closure_date, closure_reason, closure_code
    from pilot01 join wic_contact wic using (kid_id)
    left join aux.kid_ethnicity using (kid_id)
    left join last_investigations invest on wic.address_id = invest.address_id
);

\copy pilot01_contact to data/pilot/02_contact.csv with csv header;
