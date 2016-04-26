create table margin as (select k1.kid_id kid_id1, k2.kid_id kid_id2, k1.first_sample_address_id address_id, k1.date_of_birth, k1.max_bll max_bll1, k2.max_bll max_bll2 
        from output.kids k1
join output.kids k2

on k1.first_sample_address_id = k2.first_sample_address_id
and k1.kid_id != k2.kid_id
and k1.max_bll between 6 and 13
-- todo: create and use max_sample_date here... but that's only knowable after the kid's last test so technically need to revise...
and k1.first_bll6_sample_date between k2.date_of_birth - 30*9 and k2.date_of_birth + 30*12
and k1.address_count=1 and k2.address_count=1);

select max_bll1, avg(max_bll2), sum((max_bll2 >= 6)::int)/sum(1.0) bll6_prop, sum((max_bll2 >= 10)::int)/sum(1.0) bll10_prop, count(*) from margin join output.addresses using (address_id) join aux.buildings using (building_id) where date_of_birth > '2000-01-01' and units = 1 group by 1 order by 1;

-- sampled under 1yo since the bll6+ came into effect:
-- ignore lab C18 and 018 since the LOD is 5
drop table if exists under1;
create table under1 as (
    with lab as (
        select kid_id, 
            bool_and(mode_bll = 1) as lod1_always,
            bool_and(mode_bll < 5) as lod5_never
        from output.tests t join output.labs l on extract(year from t.date) = l.year and t.lab_id = l.lab_id
        group by kid_id)

    select k.* from output.kids k left join lab using (kid_id) 
    where lod5_never and 
    first_sample_date - date_of_birth < 365 and 
    first_sample_date > '2004-01-01'
);

alter table under1 add column max_bll_under1 int;

update under1 u set max_bll_under1 = b.max_bll_under1 
from (select kid_id, max(bll) max_bll_under1
        from output.tests where age <= 366 group by kid_id) b 
where u.kid_id = b.kid_id;

alter table under1 add column min_bll_after_max_bll int;
alter table under1 add column min_bll_after_max_bll_under3 int;
update under1 u set
    min_bll_after_max_bll = b.min_bll_after_max_bll,
    min_bll_after_max_bll_under3 = b.min_bll_after_max_bll_under3
from (select kid_id, 
        min(bll) min_bll_after_max_bll,
        min(CASE WHEN age < 900 THEN bll else null END) as min_bll_after_max_bll_under3
        from output.tests 
        join output.kids using (kid_id)
        where date >= max_bll_sample_date
        group by kid_id) b 
where u.kid_id = b.kid_id;

select max_bll_under1, 
    count(*) kid_count, 
    avg(greatest(max_bll,1)) as max_bll,
    avg(greatest(min_bll_after_max_bll, 1)) as min_bll,
    avg(greatest(min_bll_after_max_bll_under3, 1)) as min_bll_after_max_bll_under3,
    avg(test_count) test_count, 
    avg(last_sample_date - date_of_birth) last_sample_age
from under1 group by 1 order by 1;



-- sampled under 1yo since the bll6+ came into effect:
-- ignore lab C18 and 018 since the LOD is 5
drop table if exists under2;
create table under2 as (
    with lab as (
        select kid_id, 
            bool_and(mode_bll = 1) as lod1_always,
            bool_and(mode_bll < 5) as lod5_never
        from output.tests t join output.labs l on extract(year from t.date) = l.year and t.lab_id = l.lab_id
        group by kid_id)

    select k.* from output.kids k left join lab using (kid_id) 
    where lod5_never and 
    first_sample_date - date_of_birth < 2*365 and 
    first_sample_date > '2004-01-01'
);

alter table under2 add column max_bll_under2 int;

update under2 u set max_bll_under2 = b.max_bll_under2
from (select kid_id, max(bll) max_bll_under2
        from output.tests where age <= 2*365+5 group by kid_id) b 
where u.kid_id = b.kid_id;

alter table under2 add column min_bll_after_max_bll int;
alter table under2 add column min_bll_after_max_bll_under3 int;
update under2 u set
    min_bll_after_max_bll = b.min_bll_after_max_bll,
    min_bll_after_max_bll_under3 = b.min_bll_after_max_bll_under3
from (select kid_id, 
        min(bll) min_bll_after_max_bll,
        min(CASE WHEN age < 900 THEN bll else null END) as min_bll_after_max_bll_under3
        from output.tests 
        join output.kids using (kid_id)
        where date >= max_bll_sample_date
        group by kid_id) b 
where u.kid_id = b.kid_id;

select max_bll_under2, 
    count(*) kid_count, 
    avg(greatest(max_bll,3)) as max_bll,
    avg(greatest(min_bll_after_max_bll, 3)) as min_bll,
    avg(greatest(min_bll_after_max_bll_under3, 3)) as min_bll_after_max_bll_under3,
    avg(test_count) test_count, 
    avg(last_sample_date - date_of_birth) last_sample_age
from under2 group by 1 order by 1;

-- TODO: consolidate into one table for both under1 and under2 analysis
-- join with inspections to see discontinuity in P(I|bll)
