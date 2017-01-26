do $$
    DECLARE RISK_COUNT int := 150;
    DECLARE RISK_INSPECT int := 125;
    DECLARE BASE_COUNT int := 150;
    DECLARE BASE_INSPECT int := 125;
    DECLARE RANDOM_SEED double precision := 0;
    DECLARE DOB_MIN date := '2016-02-01'::date;
    DECLARE DOB_MAX date := '2016-09-30'::date;
BEGIN

create temp table past_pilot_addresses as (
    select address from pilot.pilot01_contact
    UNION ALL
    select address from pilot.pilot02b_contact
);

create temp table past_pilot_kids as (
    select part_id_i from pilot.pilot01
    UNION ALL
    select part_id_i from pilot.pilot02
);

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
    not (part_id_i && (select anyarray_agg(part_id_i) from past_pilot_kids))
    and not (address in (select address from past_pilot_addresses))
);
-- select risk group
create temp table pilot01_risk as (
    select *, false as inspection
    -- order by kid_id too in case there are score ties
    from wic_kids order by score desc, kid_id asc
    limit RISK_COUNT
);

-- set random addresses to receive inspection
PERFORM setseed(RANDOM_SEED);
update pilot01_risk set inspection = true 
where address_id in (
    select address_id from pilot01_risk 
    order by random() limit RISK_INSPECT
);

-- select for base group
PERFORM setseed(RANDOM_SEED);
create temp table pilot01_base as (
    select w.*, false as inspection 
    from wic_kids w
    left join aux.assessor using (address)
    join output.addresses using (address_id)
    left join aux.buildings using (building_id)
    where 
    -- kid not in risk group
    kid_id not in (select kid_id from pilot01_risk) and
    -- address not in risk group
    address_id not in (select address_id from pilot01_risk)
    -- built before 1978
    and min_age > (2014-1978) and pre1978_prop = 1
    order by random() limit BASE_COUNT
);

-- set random addresses to receive inspection
PERFORM setseed(RANDOM_SEED);
update pilot01_base set inspection = true 
where address_id in (
    select address_id from pilot01_base 
    order by random() limit BASE_INSPECT
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
    left join output.kid_ethnicity using (kid_id)
    left join last_investigations invest on wic.address_id = invest.address_id
);

\copy pilot01_contact to data/pilot/02_contact.csv with csv header;
