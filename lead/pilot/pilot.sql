do $$
    DECLARE RISK_COUNT int := 0;
    DECLARE RISK_INSPECT int := 0;
    DECLARE BASE_COUNT int := 350;
    DECLARE BASE_INSPECT int := 300;
    DECLARE RANDOM_SEED double precision := 0;
    DECLARE DOB_MIN date := '2016-02-01'::date;
    DECLARE DOB_MAX date := '2016-09-30'::date;
BEGIN

create temp table past_pilot_addresses as (
    select address from pilot.pilot01
    UNION ALL
    select address from pilot.pilot02b
    UNION ALL
    select address from pilot.pilot03
);

create temp table past_pilot_kids as (
    select part_id_i from pilot.pilot01
    UNION ALL
    select part_id_i from pilot.pilot02
    UNION ALL
    select part_id_i from pilot.pilot03
);

-- subset of predictions features for kids with eligible dob
create temp table wic_kids as (
    with kids as (
        select kid_id, score, date,
            address_id, address, first_name, last_name, date_of_birth, max_bll0 as max_bll, test_count, last_sample_date
        from pilot.predictions03
        where address_wic_min_date is not null 
            and date_of_birth 
            between DOB_MIN and DOB_MAX),
    -- get participant ids for kid
    part_id_is as (
        select kid_id, array_agg(distinct part_id_i) as part_id_i
        from pilot.predictions03
        join aux.kid_wics using (kid_id)
        group by kid_id
    )
    select kids.*, new.part_id_i from kids 
    join part_id_is new using (kid_id)
    where 
    -- exclude kids already in pilot
    not (part_id_i && (select anyarray_agg(part_id_i) from past_pilot_kids))
    -- exclude addresses already in pilot
    and not (address in (select address from past_pilot_addresses))
);
-- select risk group
create temp table pilot_risk as (
    select *, false as inspection
    -- order by kid_id too in case there are score ties
    from wic_kids order by score desc, kid_id asc
    limit RISK_COUNT
);

-- set random kids to receive inspection
-- all kids at an address are either inspected or not
-- all addresses for a kid are either inspected or not
PERFORM setseed(RANDOM_SEED);
update pilot_risk set inspection = true 
where kid_id in (
    select kid_id from (
        select address_id from pilot_risk 
        order by random() limit RISK_INSPECT)
    t join pilot_risk using (address_id) 
);

-- select for base group
PERFORM setseed(RANDOM_SEED);
create temp table pilot_base as (
    select w.*, false as inspection 
    from wic_kids w
    left join aux.assessor using (address)
    join output.addresses using (address_id)
    left join aux.buildings using (building_id)
    where 
    -- kid not in risk group
    kid_id not in (select kid_id from pilot_risk) and
    -- address not in risk group
    address_id not in (select address_id from pilot_risk)
    -- built before 1978
    and min_age > (2014-1978) and pre1978_prop = 1
    order by random() limit BASE_COUNT
);

-- set random addresses to receive inspection
PERFORM setseed(RANDOM_SEED);
update pilot_base set inspection = true 
where kid_id in (
    select kid_id from (
        select address_id from pilot_base
        order by random() limit BASE_INSPECT)
    t join pilot_base using (address_id) 
);

end $$;

-- TODO revise this?
create temp table last_investigations as (
    --select distinct on(address_id) *
    --from output.investigations
    --order by address_id, coalesce(closure_date, comply_date,init_date,referral_date) desc
    select 
        address_id, array_remove(array_agg(distinct apt), null) as investigation_apts,
        bool_or(closure_date is not null) as investigation_open,
        max(referral_date) as investigations_last_referral_date,
        max(init_date) as investigations_last_init_date,
        max(closure_date) as investigations_last_closure_date,
        max(comply_date) as investigations_last_comply_date
    from output.investigations
    group by address_id
);


drop table if exists pilot;
create table pilot as (
with pilot as (
    (select *, true as risk from pilot_risk) UNION ALL
    (select *, false as risk from pilot_base)
)

select * from
pilot
-- ethnicity for Spanish language contact
left join (select kid_id, kid_ethnicity from output.kid_ethnicity) ke using (kid_id)
left join last_investigations invest using (address_id)
);
-- copy concatenated to csv

drop table if exists pilot_contact;
create table pilot_contact as (
    select kid_id, address_id, rank() over (partition by kid_id, address_id order by last_upd_d desc, ogc_fid),
        addr_ln1_t, addr_ln2_t, addr_apt_t, cont_nme_t, relate_c, phne_nbr_n, cell_nbr_n
    from pilot 
    join wic_contact wic using (kid_id, address_id)
);
