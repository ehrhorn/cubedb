select
    meta.event_no,
    meta.level,
    meta.pid,
    meta.unix_start_time,
    meta.energy_log10,
    meta.max_size_bytes,
    meta.raw_event_length,
    meta.event_id,
    meta.sub_event_id,
    reconstruction_names.reconstruction_name,
    gcd_file_paths.gcd_file_path || '/' || gcd_file_names.gcd_file_name as gcd_files,
    i3_file_paths.i3_file_path || '/' || i3_file_names.i3_file_name || ',' || gcd_file_paths.gcd_file_path || '/' || gcd_file_names.gcd_file_name as files,
    run_types.run_type
from
    meta
    inner join reconstruction_names on meta.reconstruction = reconstruction_names.row
    inner join i3_file_paths on meta.i3_file_path = i3_file_paths.row
    inner join i3_file_names on meta.i3_file_name = i3_file_names.row
    inner join gcd_file_paths on meta.gcd_file_path = gcd_file_paths.row
    inner join gcd_file_names on meta.gcd_file_name = gcd_file_names.row
    inner join run_types on meta.run_type = run_types.row
where
    run_types.run_type in ('muongun')
    and meta.train = 1
    and abs(meta.pid) = 13
limit 100
union
select
    meta.event_no,
    meta.level,
    meta.pid,
    meta.unix_start_time,
    meta.energy_log10,
    meta.max_size_bytes,
    meta.raw_event_length,
    meta.event_id,
    meta.sub_event_id,
    reconstruction_names.reconstruction_name,
    gcd_file_paths.gcd_file_path || '/' || gcd_file_names.gcd_file_name as gcd_files,
    i3_file_paths.i3_file_path || '/' || i3_file_names.i3_file_name || ',' || gcd_file_paths.gcd_file_path || '/' || gcd_file_names.gcd_file_name as files,
    run_types.run_type
from
    meta
    inner join reconstruction_names on meta.reconstruction = reconstruction_names.row
    inner join i3_file_paths on meta.i3_file_path = i3_file_paths.row
    inner join i3_file_names on meta.i3_file_name = i3_file_names.row
    inner join gcd_file_paths on meta.gcd_file_path = gcd_file_paths.row
    inner join gcd_file_names on meta.gcd_file_name = gcd_file_names.row
    inner join run_types on meta.run_type = run_types.row
where
    run_types.run_type in ('genie')
    and meta.train = 1
    and abs(meta.pid) = 12
limit 100
union
select
    meta.event_no,
    meta.level,
    meta.pid,
    meta.unix_start_time,
    meta.energy_log10,
    meta.max_size_bytes,
    meta.raw_event_length,
    meta.event_id,
    meta.sub_event_id,
    reconstruction_names.reconstruction_name,
    gcd_file_paths.gcd_file_path || '/' || gcd_file_names.gcd_file_name as gcd_files,
    i3_file_paths.i3_file_path || '/' || i3_file_names.i3_file_name || ',' || gcd_file_paths.gcd_file_path || '/' || gcd_file_names.gcd_file_name as files,
    run_types.run_type
from
    meta
    inner join reconstruction_names on meta.reconstruction = reconstruction_names.row
    inner join i3_file_paths on meta.i3_file_path = i3_file_paths.row
    inner join i3_file_names on meta.i3_file_name = i3_file_names.row
    inner join gcd_file_paths on meta.gcd_file_path = gcd_file_paths.row
    inner join gcd_file_names on meta.gcd_file_name = gcd_file_names.row
    inner join run_types on meta.run_type = run_types.row
where
    run_types.run_type in ('genie')
    and meta.train = 1
    and abs(meta.pid) = 14
limit 100
union
select
    meta.event_no,
    meta.level,
    meta.pid,
    meta.unix_start_time,
    meta.energy_log10,
    meta.max_size_bytes,
    meta.raw_event_length,
    meta.event_id,
    meta.sub_event_id,
    reconstruction_names.reconstruction_name,
    gcd_file_paths.gcd_file_path || '/' || gcd_file_names.gcd_file_name as gcd_files,
    i3_file_paths.i3_file_path || '/' || i3_file_names.i3_file_name || ',' || gcd_file_paths.gcd_file_path || '/' || gcd_file_names.gcd_file_name as files,
    run_types.run_type
from
    meta
    inner join reconstruction_names on meta.reconstruction = reconstruction_names.row
    inner join i3_file_paths on meta.i3_file_path = i3_file_paths.row
    inner join i3_file_names on meta.i3_file_name = i3_file_names.row
    inner join gcd_file_paths on meta.gcd_file_path = gcd_file_paths.row
    inner join gcd_file_names on meta.gcd_file_name = gcd_file_names.row
    inner join run_types on meta.run_type = run_types.row
where
    run_types.run_type in ('genie')
    and meta.train = 1
    and abs(meta.pid) = 16
limit 100