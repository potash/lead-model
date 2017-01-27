create temp table contact as (
with kid_wics as (
    select kid_id, array_agg(part_id_i) part_id_i from aux.kid_wics
    group by kid_id
)

    select distinct on (kid_id, address_id) 
        wic_contact.*, part_id_i, address, 
        community_area_id, zip_code, date_of_birth, first_name, last_name,
        max_bll0 as max_bll
    from wic_contact
    join output.kids using (kid_id)
    join kid_wics using (kid_id)
    join aux.addresses using (address_id)
    where 
    -- between 1 and 2 years old 
    '2017-02-01' - date_of_birth between 366 and 2*365
    -- child's address 
    and address_wic_infant
    -- most recent entry for this child, address 
    order by kid_id, address_id, last_upd_d desc
);

drop table if exists pilot.letters;
create table pilot.letters as (
with past as (
    select part_id_i, address from pilot.pilot01_contact
    UNION ALL
    select part_id_i, address from pilot.pilot02_contact
    UNION ALL
    select part_id_i, address from pilot.pilot03
)

select *, zip_code in (60621, 60636) as englewood, false as treatment,
    coalesce(cont_nme_t, 'PARENT OR GUARDIAN OF ' || first_name || ' ' || last_name)
        AS recipient
from contact
-- address not in pilot
where address not in (select address from past) 
-- kid not in pilot
and not (part_id_i && (select anyarray_agg(part_id_i) from past))
);

select setseed(0);
update pilot.letters set treatment = true
where englewood and kid_id in (
    select kid_id from (
        select address_id from pilot.letters
        order by random() 
        limit (select count(*)/2 from pilot.letters where englewood))
    t join pilot.letters using (address_id)
);


select setseed(0);
update pilot.letters set treatment = true
where not englewood and kid_id in (
    select kid_id from (
        select address_id from pilot.letters
        order by random() 
        limit (select count(*)/2 from pilot.letters where not englewood))
    t join pilot.letters using (address_id)
);
