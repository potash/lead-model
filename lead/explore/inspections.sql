create temp table inspection_tests as (select * from output.inspections i left join output.tests k using (address_id) where i.init_date between k.date and k.date + 2*365 and bll >= 10);

create temp table inspection_kids as (select * from output.inspections i left join output.kids k on i.address_id = k.first_bll10_address_id and i.init_date between k.first_bll10_sample_date and k.first_bll10_sample_date + 2*365);
