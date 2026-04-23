
-- UFC Fight Analytics — Seed Data
-- PostgreSQL


INSERT INTO fighters (name, nickname, weight_class, stance, reach_in, height_in, wins, losses, draws, win_by_ko, win_by_sub, win_by_dec) VALUES
('Israel Adesanya',  'The Last Stylebender', 'Middleweight',      'Switch',    80, 76, 24, 3, 0, 16, 1,  7),
('Alex Pereira',     'Poatan',               'Light Heavyweight', 'Orthodox',  79, 76, 12, 2, 0,  9, 0,  3),
('Jon Jones',        'Bones',                'Heavyweight',       'Orthodox',  84, 76, 27, 1, 0, 10, 5, 12),
('Islam Makhachev',  NULL,                   'Lightweight',       'Orthodox',  70, 70, 26, 1, 0,  6, 8, 12),
('Leon Edwards',     'Rocky',                'Welterweight',      'Orthodox',  74, 72, 22, 3, 0,  9, 2, 11),
('Dricus Du Plessis','Stillknocks',          'Middleweight',      'Orthodox',  76, 72, 22, 2, 0, 13, 4,  5),
('Khamzat Chimaev',  'Borz',                 'Middleweight',      'Orthodox',  77, 72, 13, 0, 0,  6, 3,  4),
('Aljamain Sterling','Funk Master',          'Bantamweight',      'Orthodox',  71, 67, 23, 4, 0,  7, 9,  7),
('Charles Oliveira', 'Do Bronx',             'Lightweight',       'Orthodox',  74, 70, 34, 9, 0,  8,21,  5),
('Dustin Poirier',   'Diamond',              'Lightweight',       'Orthodox',  72, 69, 30, 8, 0, 15, 6,  9);

INSERT INTO events (event_name, event_date, location, card_type) VALUES
('UFC 287',           '2023-04-08', 'Miami, FL',       'PPV'),
('UFC 295',           '2023-11-11', 'New York, NY',    'PPV'),
('UFC 300',           '2024-04-13', 'Las Vegas, NV',   'UFC 300'),
('UFC Fight Night 231','2023-09-09','Las Vegas, NV',   'Fight Night'),
('UFC 302',           '2024-06-01', 'Newark, NJ',      'PPV');

INSERT INTO fights (event_id, fighter1_id, fighter2_id, winner_id, weight_class, scheduled_rounds, actual_rounds, win_method, win_round, win_time, is_title_fight) VALUES
(1, 1, 6,  6, 'Middleweight',      5, 5, 'Decision - Unanimous', 5, '5:00', TRUE),
(2, 2, 3,  2, 'Light Heavyweight', 5, 5, 'Decision - Unanimous', 5, '5:00', TRUE),
(3, 2, 1,  2, 'Light Heavyweight', 5, 2, 'KO/TKO',               2, '4:02', TRUE),
(4, 4, 9,  4, 'Lightweight',       5, 5, 'Decision - Unanimous', 5, '5:00', TRUE),
(5, 4, 10,10, 'Lightweight',       5, 5, 'Decision - Unanimous', 5, '5:00', FALSE);

-- Fight 1: Adesanya vs Du Plessis — 5 rounds
INSERT INTO round_stats (fight_id, fighter_id, round_number, sig_strikes_landed, sig_strikes_attempted, total_strikes_landed, head_strikes_landed, body_strikes_landed, leg_strikes_landed, takedowns_landed, takedowns_attempted, ctrl_time_seconds, knockdowns) VALUES
(1,1,1, 18,32,22,10,4,4, 0,0, 10,0),
(1,6,1, 14,26,17, 8,3,3, 1,2, 55,0),
(1,1,2, 22,38,27,13,5,4, 0,1,  8,0),
(1,6,2, 19,33,24,11,4,4, 2,3, 80,0),
(1,1,3, 16,28,20, 9,3,4, 0,0, 12,0),
(1,6,3, 21,35,26,12,5,4, 1,2, 65,0),
(1,1,4, 24,41,30,15,6,3, 0,0, 15,1),
(1,6,4, 17,30,21,10,4,3, 1,2, 45,0),
(1,1,5, 20,34,25,12,4,4, 0,0, 20,0),
(1,6,5, 22,38,28,13,5,4, 1,2, 70,1);

-- Fight 2: Pereira vs Jones — 5 rounds
INSERT INTO round_stats (fight_id, fighter_id, round_number, sig_strikes_landed, sig_strikes_attempted, total_strikes_landed, head_strikes_landed, body_strikes_landed, leg_strikes_landed, takedowns_landed, takedowns_attempted, ctrl_time_seconds, knockdowns) VALUES
(2,2,1, 21,35,25,13,5,3, 0,0, 18,0),
(2,3,1, 16,28,20, 9,4,3, 1,2, 60,0),
(2,2,2, 28,44,34,17,8,3, 0,1, 15,1),
(2,3,2, 14,26,18, 8,3,3, 2,3, 95,0),
(2,2,3, 19,32,24,12,4,3, 0,0, 12,0),
(2,3,3, 18,31,23,10,5,3, 1,2, 80,0),
(2,2,4, 25,40,31,15,7,3, 0,0, 20,1),
(2,3,4, 15,27,19, 9,3,3, 1,2, 55,0),
(2,2,5, 22,37,28,14,6,2, 0,0, 25,0),
(2,3,5, 17,30,22,10,4,3, 1,2, 70,0);

-- Fight 3: Pereira KO Adesanya — 2 rounds
INSERT INTO round_stats (fight_id, fighter_id, round_number, sig_strikes_landed, sig_strikes_attempted, total_strikes_landed, head_strikes_landed, body_strikes_landed, leg_strikes_landed, takedowns_landed, takedowns_attempted, ctrl_time_seconds, knockdowns) VALUES
(3,2,1, 19,31,24,12,5,2, 0,0, 22,1),
(3,1,1, 22,37,27,14,5,3, 0,0, 18,0),
(3,2,2, 24,38,29,15,6,3, 0,0, 15,1),
(3,1,2, 15,28,19, 9,3,3, 0,0, 10,0);

-- Fight 4: Makhachev vs Oliveira — 5 rounds
INSERT INTO round_stats (fight_id, fighter_id, round_number, sig_strikes_landed, sig_strikes_attempted, total_strikes_landed, head_strikes_landed, body_strikes_landed, leg_strikes_landed, takedowns_landed, takedowns_attempted, submission_attempts, ctrl_time_seconds, knockdowns) VALUES
(4,4,1, 12,22,18, 6,3,3, 2,3,0,125,0),
(4,9,1, 16,28,20,10,4,2, 0,1,2, 30,0),
(4,4,2, 14,24,19, 7,4,3, 3,4,0,155,0),
(4,9,2, 18,32,22,11,5,2, 0,1,3, 20,0),
(4,4,3, 16,26,21, 8,5,3, 2,3,0,130,0),
(4,9,3, 20,35,25,12,6,2, 0,0,1, 25,0),
(4,4,4, 10,18,14, 5,3,2, 1,2,0,145,0),
(4,9,4, 22,38,27,14,6,2, 0,0,2, 15,0),
(4,4,5, 13,22,17, 7,4,2, 2,3,0,120,0),
(4,9,5, 17,30,21,10,5,2, 0,1,1, 25,0);

-- Fight 5: Makhachev vs Poirier — 5 rounds
INSERT INTO round_stats (fight_id, fighter_id, round_number, sig_strikes_landed, sig_strikes_attempted, total_strikes_landed, head_strikes_landed, body_strikes_landed, leg_strikes_landed, takedowns_landed, takedowns_attempted, ctrl_time_seconds, knockdowns) VALUES
(5, 4,1, 11,21,15, 5,3,3, 2,3,105,0),
(5,10,1, 20,34,25,13,5,2, 0,0, 20,0),
(5, 4,2, 13,23,17, 6,4,3, 1,3,115,0),
(5,10,2, 24,40,30,15,7,2, 0,0, 18,1),
(5, 4,3, 10,20,14, 5,3,2, 2,3,120,0),
(5,10,3, 22,37,28,14,6,2, 0,0, 22,0),
(5, 4,4,  9,18,12, 4,3,2, 1,2,100,0),
(5,10,4, 26,42,32,16,8,2, 0,0, 20,1),
(5, 4,5, 11,20,14, 5,3,3, 1,2, 95,0),
(5,10,5, 21,35,26,13,6,2, 0,0, 25,0);
