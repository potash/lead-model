create temp table letters as (
with

-- addresses with kids under 1
addresses_under1 as (select address_id from output.kid_addresses join output.kids using (kid_id) where first_wic_date is not null and date_of_birth >= ('2016-04-26' - interval '1 year')),

-- all kids between 1 and 2
kids1to2 as (select kid_id, date_of_birth from output.kids where first_wic_date is not null and date_of_birth between ('2016-04-26' - interval '2 year') and ('2016-04-26' - interval '1 year')),

-- exclude kids sharing address with kids under 1
kids1to2_exclude as (select kid_id from kids1to2 join output.kid_addresses using (kid_id) join addresses_under1 using (address_id)),

-- include the rest of the kids
kids1to2_include as (select kid_id, date_of_birth from kids1to2 where kid_id not in (select kid_id from kids1to2_exclude)),

letters as (
select distinct on (kid_id) kid_id, part_id_i,
    brth_lst_t,
    brth_fst_t,
    birth_d,
    a.last_upd_d,
    addr_ln1_t,
    addr_ln2_t,
    addr_apt_t,
    addr_cty_t,
    addr_st_c,
    addr_zip_n,
    zip_ext_n,
    cont_nme_t,
    relate_c
--    ,
--    county_c,
--    phne_nbr_n,
--    cell_nbr_n

from kids1to2_include
join aux.kid_wics using (kid_id)
join cornerstone.partenrl using (part_id_i)
join cornerstone.partaddr a on part_id_i = addr_id_i
order by kid_id, a.last_upd_d desc
)

select * from letters where cont_nme_t is not null
);

\copy letters to data/pilot/letters.csv with csv header;
