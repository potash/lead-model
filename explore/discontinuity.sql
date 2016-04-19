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
