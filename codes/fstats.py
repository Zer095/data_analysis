import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import os

CATEGORIES = ['GK', 'DF', 'MF', 'FW']
EPOCHS = 100
TEST_SIZE = 0.4

def scrap():

    # Get standard stats from the website-----------------------------------------------------------------
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Geck0) Chrome/47.0.2526.106 Safari/537.36' }
    standard_page = "https://fbref.com/en/comps/11/stats/Serie-A-Stats"
    pageTree = requests.get(standard_page, headers=headers)
    # Remove comments
    html = pageTree.content.decode("utf-8").replace("<!--","").replace("-->","")
    # Parse the page
    altSoup = BeautifulSoup(html, 'html.parser')

    # General data------------------------------------------------------------------------------------------

    # Players' name
    players = altSoup.find_all("td", {"data-stat": "player"})
    # Players' nationality
    nationality = altSoup.find_all("td", {"data-stat": "nationality"})
    # Players' positions
    positions = altSoup.find_all("td", {"data-stat": "position"})
    # Players' teams
    teams = altSoup.find_all("td", {"data-stat": "team"})
    # Players' birth years
    years = altSoup.find_all("td", {"data-stat": "birth_year"})

    # Correct the lenght of the next statistics
    n = len(players)
    # Players' played games 
    games = altSoup.find_all("td", {"data-stat": "games"})
    del games[0: (len(games)-n)]
    # Players' started games
    start_games = altSoup.find_all("td", {"data-stat": "games_starts" })
    del start_games[0: (len(start_games)-n)]
    # Players' played minutes
    minutes = altSoup.find_all("td", {"data-stat": "minutes"})
    del minutes[0: (len(minutes)-n)]
    # Players' goals
    goals = altSoup.find_all("td", {"data-stat":"goals"})
    del goals[0: (len(goals)-n)]
    # Players' assists
    assists = altSoup.find_all("td", {"data-stat":"assists"})
    del assists[0: (len(assists)-n)]
    # Players' non-penalty goals
    non_penalty = altSoup.find_all("td", {"data-stat":"goals_pens"})
    del non_penalty[0: (len(non_penalty)-n)]
    # Players' penalty made
    penalty_made = altSoup.find_all("td", {"data-stat":"pens_made"})
    del penalty_made[0: (len(penalty_made)-n)]
    # Players' penalty attempted
    penalty_att = altSoup.find_all("td", {"data-stat":"pens_att"})
    del penalty_att[0: (len(penalty_att)-n)]
    # Players' Expected Goals
    xG = altSoup.find_all("td", {"data-stat":"xg"} )
    del xG[0: (len(xG)-n)]
    # Players' non penalty expected goals
    npxG = altSoup.find_all("td", {"data-stat": "npxg"})
    del npxG[0: (len(npxG)-n)]
    # Players' expected assist
    xAG = altSoup.find_all("td", {"data-stat":"xg_assist"})
    del xAG[0: (len(xAG)-n)]
    # Players' progressive carries
    prgC = altSoup.find_all("td", {"data-stat":"progressive_carries"})
    del prgC[0: (len(prgC)-n)]
    # Players' progressive passes
    prgP = altSoup.find_all("td", {"data-stat":"progressive_passes"})
    del prgP[0: (len(prgP)-n)]
    # Players' progressive passes received
    prgR = altSoup.find_all("td", {"data-stat":"progressive_passes_received"})
    del prgR[0: (len(prgR)-n)]

    for i in range(n):
        players[i] = str(players[i]).split("<a")[1].split("</a")[0].split(">")[1]
        nationality[i] = str(nationality[i]).split("/country/")[1].split('">')[0].split("/")[0]
        positions[i] = str(positions[i]).split("</td")[0].split(">")[1]
        teams[i] = str(teams[i]).split("</a")[0].split('s">')[1]
        years[i] = int(str(years[i]).split("</td")[0].split('">')[1])
        games[i] = int(str(games[i]).split("</td")[0].split('">')[1])
        start_games[i] = int(str(start_games[i]).split("</td")[0].split('">')[1])
        minutes[i] = int(str(minutes[i]).split("</td")[0].split('">')[1].replace(",",""))
        goals[i] = int(str(goals[i]).split("</td")[0].split('">')[1])
        assists[i] = int(str(assists[i]).split("</td")[0].split('">')[1])
        non_penalty[i] = int(str(non_penalty[i]).split("</td")[0].split('">')[1])
        penalty_made[i] = int(str(penalty_made[i]).split("</td")[0].split('">')[1])
        penalty_att[i] = int(str(penalty_att[i]).split("</td")[0].split('">')[1])
        try:
            xG[i] = float(str(xG[i]).split("</td")[0].split('">')[1])
        except ValueError:
            xG[i] = 0
        try:
            npxG[i] = float(str(npxG[i]).split("</td")[0].split('">')[1])
        except ValueError:
            npxG[i] = 0
        try:
            xAG[i] = float(str(xAG[i]).split("</td")[0].split('">')[1])
        except ValueError:
            xAG[i] = 0
        try:
            prgC[i] = int(str(prgC[i]).split("</td")[0].split('">')[1])
        except ValueError:
            prgC[i] = 0
        try:
            prgP[i] = int(str(prgP[i]).split("</td")[0].split('">')[1])
        except ValueError:
            prgP[i] = 0
        try:
            prgR[i] = int(str(prgR[i]).split("</td")[0].split('">')[1])
        except ValueError:
            prgR[i] = 0

    # Get shooting stats from the website-----------------------------------------------------------------
    shooting_page = "https://fbref.com/en/comps/11/shooting/Serie-A-Stats"
    shootingTree = requests.get(shooting_page, headers=headers)
    shooting_html = shootingTree.content.decode("utf-8").replace("<!--","").replace("-->","")
    shootingSoup = BeautifulSoup(shooting_html, 'html.parser')

    # Players' total shots
    shots = shootingSoup.find_all("td", {"data-stat":"shots"})
    del shots[0:(len(shots)-n)]
    # Players' shots on target
    shots_target = shootingSoup.find_all("td", {"data-stat":"shots_on_target"})
    del shots_target[0:(len(shots_target)-n)]
    # Players' goal per shots
    goal_per_shots = shootingSoup.find_all("td", {"data-stat":"goals_per_shot"})
    del goal_per_shots[0:(len(goal_per_shots)-n)]
    # Players goal per shots on target
    goal_shots_target = shootingSoup.find_all("td", {"data-stat":"goals_per_shot_on_target"})
    del goal_shots_target[0: len(goal_shots_target) - n]
    # Players' average shot distance
    shot_distance = shootingSoup.find_all("td", {"data-stat":"average_shot_distance"})
    del shot_distance[0:len(shot_distance)-n]

    # Store shooting statistics
    for i in range(n):
        shots[i] = int(str(shots[i]).split("</td")[0].split('">')[1])
        shots_target[i] = int(str(shots_target[i]).split("</td")[0].split('">')[1])
        try:
            goal_per_shots[i] = int(str(goal_per_shots[i]).split("</td")[0].split('">')[1])
        except ValueError:
            goal_per_shots[i] = 0
        try:
            goal_shots_target[i] = int(str(goal_shots_target[i]).split("</td")[0].split('">')[1])
        except ValueError:
            goal_shots_target[i] = 0
        try:
            shot_distance[i] = int(str(shot_distance[i]).split("</td")[0].split('">')[1])
        except ValueError:
            shot_distance[i] = 0


    # Get passing stats from the website------------------------------------------------------
    passing_page = "https://fbref.com/en/comps/11/passing/Serie-A-Stats"
    passingTree = requests.get(passing_page, headers=headers)
    passing_html = passingTree.content.decode("utf-8").replace("<!--","").replace("-->","")
    passingSoup = BeautifulSoup(passing_html, 'html.parser')

    # Players' completed passes
    completed_passes = passingSoup.find_all("td", {"data-stat":"passes_completed"})
    del completed_passes[0: len(completed_passes)-n]
    # Players' attempted passes
    attempted_passes = passingSoup.find_all("td", {"data-stat":"passes"})
    del attempted_passes[0:len(attempted_passes)-n]
    # Players' completition passes
    completed_perc = passingSoup.find_all("td", {"data-stat":"passes_pct"})
    del completed_perc[0:len(completed_perc)-n]
    # Players' total passing distance
    passing_distance = passingSoup.find_all("td", {"data-stat":"passes_total_distance"})
    del passing_distance[0:len(passing_distance)-n]
    # Players' progressive passing distance
    prog_pass_distance = passingSoup.find_all("td", {"data-stat":"passes_progressive_distance"})
    del prog_pass_distance[0:len(prog_pass_distance)-n]
    # Players' completed short passes
    short_passes_completed = passingSoup.find_all("td", {"data-stat":"passes_completed_short"})
    del short_passes_completed[0: len(short_passes_completed)-n]
    # Players' attempted short passes
    short_passes_attempted = passingSoup.find_all("td", {"data-stat":"passes_short"})
    del short_passes_attempted[0:len(short_passes_attempted)-n]
    # Players' completed medium passes
    medium_passes_completed = passingSoup.find_all("td", {"data-stat":"passes_completed_medium"})
    del medium_passes_completed[0:len(medium_passes_completed)-n]
    # Players' attempted medium passes
    medium_passes_attempted = passingSoup.find_all("td", {"data-stat":"passes_medium"})
    del medium_passes_attempted[0:len(medium_passes_attempted)-n]
    # Players' completed long passes
    long_passes_completed = passingSoup.find_all("td", {"data-stat":"passes_long"})
    del long_passes_completed[0:len(long_passes_completed)-n]
    # Players' attempted long passes
    long_passes_attempted = passingSoup.find_all("td", {"data-stat":"passes_long"})
    del long_passes_attempted[0:len(long_passes_attempted)-n]
    # Players' key passes
    key_passes = passingSoup.find_all("td", {"data-stat":"assisted_shots"})
    del key_passes[0:len(key_passes)-n]
    # Players' passes into final_third
    passes_1t = passingSoup.find_all("td", {"data-stat":"passes_into_final_third"})
    del passes_1t[0:len(passes_1t)-n]
    # Players' passes into the penalty area
    ppa = passingSoup.find_all("td", {"data-stat":"passes_into_penalty_area"})
    del ppa[0:len(ppa)-n]
    # Players' crosses into penalty area
    cpa = passingSoup.find_all("td", {"data-stat":"crosses_into_penalty_area"})
    del cpa[0:len(cpa)-n]

    # Store data
    for i in range(n):
        try:
            completed_passes[i] = int(str(completed_passes[i]).split("</td")[0].split('">')[1])
        except ValueError:
            completed_passes[i]
        try:
            attempted_passes[i] = int(str(attempted_passes[i]).split("</td")[0].split('">')[1])
        except ValueError:
            attempted_passes[i] = 0
        try:
            completed_perc[i] = int(str(completed_perc[i]).split("</td")[0].split('">')[1].replace(".",""))
        except ValueError:
            completed_perc[i] = 0
        try:
            passing_distance[i] = int(str(passing_distance[i]).split("</td")[0].split('">')[1])
        except ValueError:
            passing_distance[i] = 0
        try:
            prog_pass_distance[i] = int(str(prog_pass_distance[i]).split("</td")[0].split('">')[1])
        except ValueError:
            prog_pass_distance[i] = 0
        try:
            short_passes_completed[i] = int(str(short_passes_completed[i]).split("</td")[0].split('">')[1])
        except ValueError:
            short_passes_completed[i] = 0
        try:
            short_passes_attempted[i] = int(str(short_passes_attempted[i]).split("</td")[0].split('">')[1])
        except ValueError:
            short_passes_attempted[i] = 0
        try:
            medium_passes_completed[i] = int(str(medium_passes_completed[i]).split("</td")[0].split('">')[1])
        except ValueError:
            medium_passes_completed[i] = 0
        try:
            medium_passes_attempted[i] = int(str(medium_passes_attempted[i]).split("</td")[0].split('">')[1])
        except ValueError:
            medium_passes_attempted[i] = 0
        try:
            long_passes_completed[i] = int(str(long_passes_completed[i]).split("</td")[0].split('">')[1])
        except ValueError:
            long_passes_completed[i] = 0
        try:
            long_passes_attempted[i] = int(str(long_passes_attempted[i]).split("</td")[0].split('">')[1])
        except ValueError:
            long_passes_attempted[i] = 0
        try:
            key_passes[i] = int(str(key_passes[i]).split("</td")[0].split('">')[1])
        except ValueError:
            key_passes[i] = 0
        try:
            passes_1t[i] = int(str(passes_1t[i]).split("</td")[0].split('">')[1])
        except ValueError:
            passes_1t[i] = 0
        try:
            ppa[i] = int(str(ppa[i]).split("</td")[0].split('">')[1])
        except ValueError:
            ppa[i] = 0
        try:
            cpa[i] = int(str(cpa[i]).split("</td")[0].split('">')[1])
        except ValueError:
            cpa[i] = 0

    # Get passing types from the website----------------------------------------------------------------------
    type_page = "https://fbref.com/en/comps/11/passing_types/Serie-A-Stats"
    typeTree = requests.get(type_page, headers=headers)
    type_html = typeTree.content.decode("utf-8").replace("<!--","").replace("-->","")
    typeSoup = BeautifulSoup(type_html, 'html.parser')

    # Players' live ball passes
    live_passes = typeSoup.find_all("td", {"data-stat":"passes_live"})
    del live_passes[0:len(live_passes)-n]
    # Player's dead ball passes
    dead_passes = typeSoup.find_all("td",{"data-stat":"passes_dead"})
    del dead_passes[0:len(dead_passes)-n]
    # Player's free kick passes
    FK_passes = typeSoup.find_all("td",{"data-stat":"passes_free_kicks"})
    del FK_passes[0:len(FK_passes)-n]
    # Players' through ball passes
    through_passes = typeSoup.find_all("td",{"data-stat":"through_balls"})
    del through_passes[0:len(through_passes)-n]
    # Players' switches
    switches = typeSoup.find_all("td", {"data-stat":"passes_switches"})
    del switches[0:len(switches)-n]
    # Players' crosses
    crosses = typeSoup.find_all("td", {"data-stat":"crosses"})
    del crosses[0:len(crosses)-n]
    # Players' throw-ins taken
    throw_in = typeSoup.find_all("td", {"data-stat":"throw_ins"})
    del throw_in[0:len(throw_in)-n]
    
    for i in range(n):
        try:
            live_passes[i] = int(str(live_passes[i]).split("</td")[0].split('">')[1])
        except ValueError:
            live_passes[i] = 0
        try:
            dead_passes[i] = int(str(dead_passes[i]).split("</td")[0].split('">')[1])
        except ValueError:
            dead_passes[i] = 0
        try:
            FK_passes[i] = int(str(FK_passes[i]).split("</td")[0].split('">')[1])
        except ValueError:
            FK_passes[i] = 0
        try:
            through_passes[i] = int(str(through_passes[i]).split("</td")[0].split('">')[1])
        except ValueError:
            through_passes[i]
        try:
            switches[i] = int(str(switches[i]).split("</td")[0].split('">')[1])
        except ValueError:
            switches[i] = 0
        try:
            crosses[i] = int(str(crosses[i]).split("</td")[0].split('">')[1])
        except ValueError:
            crosses[i] = 0
        try:
            throw_in[i] = int(str(throw_in[i]).split("</td")[0].split('">')[1])
        except ValueError:
            throw_in[i] = 0


    # Get Goal and Shot creation stats----------------------------------------------------
    creation_page = "https://fbref.com/en/comps/11/gca/Serie-A-Stats"
    creationTree = requests.get(creation_page,headers=headers)
    creation_html = creationTree.content.decode("utf-8").replace("<!--","").replace("-->","")
    creationSoup = BeautifulSoup(creation_html, 'html.parser')

    # Players' shot creating actio
    sca =  creationSoup.find_all("td", {"data-stat":"sca"})
    del sca[0:len(sca)-n]
    # Players' live passes that lead to a shot attempt
    sca_live = creationSoup.find_all("td", {"data-stat":"sca_passes_live"})
    del sca_live[0:len(sca_live)-n]
    # Players' dead passes that lead to a shot attempt
    sca_dead = creationSoup.find_all("td", {"data-stat":"sca_passes_dead"})
    del sca_dead[0:len(sca_dead)-n]
    # Players' successful take-ons that lead to a shot attempt
    sca_to = creationSoup.find_all("td",{"data-stat":"sca_take_ons"})
    del sca_to[0:len(sca_to)-n]
    # Players' shot attempts that lead to another shot attempt
    sca_sh = creationSoup.find_all("td",{"data-stat":"sca_shots"})
    del sca_sh[0:len(sca_sh)-n]
    # Players' defensive actions that lead to a shot attempt
    sca_def = creationSoup.find_all("td",{"data-stat":"sca_defense"})
    del sca_def[0:len(sca_def)-n]
    # Players' goal creating actions
    gca = creationSoup.find_all("td",{"data-stat":"gca"})
    del gca[0:len(gca)-n]
    # Players' live passes that lead to a goal
    gca_live = creationSoup.find_all("td", {"data-stat":"gca_passes_live"})
    del gca_live[0:len(gca_live)-n]
    # Players' dead passes that lead to a goal
    gca_dead = creationSoup.find_all("td", {"data-stat":"gca_passes_dead"})
    del gca_dead[0:len(gca_dead)-n]
    # Players' successful take-ons that lead to a goal
    gca_to = creationSoup.find_all("td",{"data-stat":"gca_take_ons"})
    del gca_to[0:len(gca_to)-n]
    # Players' shot attempts that lead to a goal
    gca_sh = creationSoup.find_all("td",{"data-stat":"gca_shots"})
    del gca_sh[0:len(gca_sh)-n]
    # Players' defensive actions that lead to a goal
    gca_def = creationSoup.find_all("td",{"data-stat":"gca_defense"})
    del gca_def[0:len(gca_def)-n]

    for i in range(n):
        try:
            sca[i] = int(str(sca[i]).split("</td")[0].split('">')[1])
        except ValueError:
            sca[i] = 0
        try:
            sca_live[i] = int(str(sca_live[i]).split("</td")[0].split('">')[1])
        except ValueError:
            sca_live[i] = 0
        try:
            sca_dead[i] = int(str(sca_dead[i]).split("</td")[0].split('">')[1])
        except ValueError:
            sca_dead[i] = 0
        try:
            sca_to[i] = int(str(sca_to[i]).split("</td")[0].split('">')[1])
        except ValueError:
            sca_to[i] = 0
        try:
            sca_sh[i] = int(str(sca_sh[i]).split("</td")[0].split('">')[1])
        except ValueError:
            sca_sh[i] = 0
        try:
            sca_def[i] = int(str(sca_def[i]).split("</td")[0].split('">')[1])
        except ValueError:
            sca_def[i] = 0
        try:
            gca[i] = int(str(gca[i]).split("</td")[0].split('">')[1])
        except ValueError:
            gca[i] = 0
        try:
            gca_live[i] = int(str(gca_live[i]).split("</td")[0].split('">')[1])
        except ValueError:
            gca_live[i] = 0
        try:
            gca_dead[i] = int(str(gca_dead[i]).split("</td")[0].split('">')[1])
        except ValueError:
            gca_dead[i] = 0
        try:
            gca_to[i] = int(str(gca_to[i]).split("</td")[0].split('">')[1])
        except ValueError:
            gca_to[i] = 0
        try:
            gca_sh[i] = int(str(gca_sh[i]).split("</td")[0].split('">')[1])
        except ValueError:
            gca_sh[i] = 0
        try:
            gca_def[i] = int(str(gca_def[i]).split("</td")[0].split('">')[1])
        except ValueError:
            gca_def[i] = 0

    # Get defensive stats
    defensive_page = "https://fbref.com/en/comps/11/defense/Serie-A-Stats"
    defensiveTree = requests.get(defensive_page, headers=headers)
    defensive_html = defensiveTree.content.decode("utf-8").replace("<!--","").replace("-->","")
    defensiveSoup = BeautifulSoup(defensive_html, 'html.parser')

    # Players' tackles
    tackles = defensiveSoup.find_all("td",{"data-stat":"tackles"})
    del tackles[0:len(tackles)-n]
    # Players' tackles won
    tackles_won = defensiveSoup.find_all("td",{"data-stat":"tackles_won"})
    del tackles_won[0:len(tackles_won)-n]
    # Players' tackles in defensive 1/3
    tackles_def_3rd = defensiveSoup.find_all("td",{"data-stat":"tackles_def_3rd"})
    del tackles_def_3rd[0:len(tackles_def_3rd)-n]
    # Players' tackles in mid third
    tackles_mid_3rd = defensiveSoup.find_all("td",{"data-stat":"tackles_mid_3rd"})
    del tackles_mid_3rd[0:len(tackles_mid_3rd)-n]
    # Players' tackles in att third
    tackles_att_3rd = defensiveSoup.find_all("td",{"data-stat":"tackles_att_3rd"})
    del tackles_att_3rd[0:len(tackles_att_3rd)-n]
    # Players' dribblers tackled
    tackles_drib = defensiveSoup.find_all("td",{"data-stat":"challenge_tackles"})
    del tackles_drib[0:len(tackles_drib)-n]
    # Players
    tackles_challenges = defensiveSoup.find_all("td",{"data-stat":"challenges"})
    del tackles_challenges[0:len(tackles_challenges)-n]
    # Challenges lost
    tackles_lost = defensiveSoup.find_all("td",{"data-stat": "challenges_lost"})
    del tackles_lost[0:len(tackles_lost)-n]
    # Blocks
    blocks = defensiveSoup.find_all("td",{"data-stat": "blocks"})
    del blocks[0:len(blocks)-n]
    # Blocked shots
    block_shots = defensiveSoup.find_all("td",{"data-stat": "blocked_shots"})
    del block_shots[0:len(block_shots)-n]
    # Blocked passes
    block_passes = defensiveSoup.find_all("td",{"data-stat": "blocked_passes"})
    del block_passes[0:len(block_passes)-n]
    # Interceptions
    interceptions = defensiveSoup.find_all("td",{"data-stat": "interceptions"})
    del interceptions[0:len(interceptions)-n]
    # Clearences
    clearences = defensiveSoup.find_all("td",{"data-stat": "clearances"})
    del clearences[0:len(clearences)-n]
    # Errors
    errors = defensiveSoup.find_all("td",{"data-stat": "errors"})
    del errors[0:len(errors)-n]

    for i in range(n):
        try:
            tackles[i] = int(str(tackles[i]).split("</td")[0].split('">')[1])
        except ValueError:
            tackles[i] = 0
        try:
            tackles_won[i] = int(str(tackles_won[i]).split("</td")[0].split('">')[1])
        except ValueError:
            tackles_won[i] = 0
        try:    
            tackles_def_3rd[i] = int(str(tackles_def_3rd[i]).split("</td")[0].split('">')[1])
        except ValueError:
            tackles_def_3rd[i] = 0
        try:
            tackles_mid_3rd[i] = int(str(tackles_mid_3rd[i]).split("</td")[0].split('">')[1])
        except ValueError:
            tackles_mid_3rd[i] = 0
        try:
            tackles_att_3rd[i] = int(str(tackles_att_3rd[i]).split("</td")[0].split('">')[1])
        except ValueError:
            tackles_att_3rd[i] = 0
        try:
            tackles_drib[i] = int(str(tackles_drib[i]).split("</td")[0].split('">')[1])
        except ValueError:
            tackles_drib[i] = 0
        try:
            tackles_challenges[i] = int(str(tackles_challenges[i]).split("</td")[0].split('">')[1])
        except ValueError:
            tackles_challenges[i] = 0
        try:
            tackles_lost[i] = int(str(tackles_lost[i]).split("</td")[0].split('">')[1])
        except ValueError:
            tackles_lost[i] = 0
        try:
            blocks[i] = int(str(blocks[i]).split("</td")[0].split('">')[1])
        except ValueError:
            blocks[i] = 0
        try:
            block_shots[i] = int(str(block_shots[i]).split("</td")[0].split('">')[1])
        except ValueError:
            block_shots[i] = 0
        try:
            block_passes[i] = int(str(block_passes[i]).split("</td")[0].split('">')[1])
        except ValueError:
            block_passes[i] = 0
        try:
            interceptions[i] = int(str(interceptions[i]).split("</td")[0].split('">')[1])
        except ValueError:
            interceptions[i] = 0
        try:
            clearences[i] = int(str(clearences[i]).split("</td")[0].split('">')[1])
        except ValueError:
            clearences[i] = 0
        try:
            errors[i] = int(str(errors[i]).split("</td")[0].split('">')[1])
        except ValueError:
            errors[i] = 0


    # Get possession stats
    possession_page = "https://fbref.com/en/comps/11/possession/Serie-A-Stats"
    possessionTree = requests.get(possession_page, headers=headers)
    possession_html = possessionTree.content.decode("utf-8").replace("<!--","").replace("-->","")
    possessionSoup = BeautifulSoup(possession_html, 'html.parser')


    # Touches
    touches = possessionSoup.find_all("td",{"data-stat":"touches"})
    del touches[0:len(touches)-n]
    # Touches in the defensive penalty area
    touches_def_pen = possessionSoup.find_all("td",{"data-stat":"touches_def_pen_area"})
    del touches_def_pen[0:len(touches_def_pen)-n]
    # Touches in the defensive 1/3
    touches_def_3rd = possessionSoup.find_all("td",{"data-stat":"touches_def_3rd"})
    del touches_def_3rd[0:len(touches_def_3rd)-n]
    # Touches in the mid 1/3
    touches_mid_3rd = possessionSoup.find_all("td",{"data-stat":"touches_mid_3rd"})
    del touches_mid_3rd[0:len(touches_mid_3rd)-n]
    # Touches in the att 1/3
    touches_att_3rd = possessionSoup.find_all("td",{"data-stat":"touches_att_3rd"})
    del touches_att_3rd[0:len(touches_att_3rd)-n]    
    # Touches in the attacking penalty area
    touches_att_pen = possessionSoup.find_all("td",{"data-stat":"touches_att_pen_area"})
    del touches_att_pen[0:len(touches_att_pen)-n]
    # Live ball touches
    touches_live_ball = possessionSoup.find_all("td",{"data-stat":"touches_live_ball"})
    del touches_live_ball[0:len(touches_live_ball)-n]
    # Take-ons attempted
    take_ons = possessionSoup.find_all("td",{"data-stat":"take_ons"})
    del take_ons[0:len(take_ons)-n]
    # Successful take-ons
    take_ons_won = possessionSoup.find_all("td",{"data-stat":"take_ons_won"})
    del take_ons_won[0:len(take_ons_won)-n]
    # Take-ons tackled
    take_ons_tackled = possessionSoup.find_all("td",{"data-stat":"take_ons_tackled"})
    del take_ons_tackled[0:len(take_ons_tackled)-n]
    # Carries
    carries = possessionSoup.find_all("td",{"data-stat":"carries"})
    del carries[0:len(carries)-n]
    # Total distance of carries
    carries_dist = possessionSoup.find_all("td",{"data-stat":"carries_distance"})
    del carries_dist[0:len(carries_dist)-n]
    # Progressive carrying distance
    carries_prog_dist = possessionSoup.find_all("td",{"data-stat":"carries_progressive_distance"})
    del carries_prog_dist[0:len(carries_prog_dist)-n]
    # Progressive carries
    carries_prog = possessionSoup.find_all("td",{"data-stat":"progressive_carries"})
    del carries_prog[0:len(carries_prog)-n]
    # Carries into the final third
    carries_fin_3rd = possessionSoup.find_all("td",{"data-stat":"carries_into_final_third"})
    del carries_fin_3rd[0:len(carries_fin_3rd)-n]
    # Carries into the penalty area
    carries_pen = possessionSoup.find_all("td",{"data-stat":"carries_into_penalty_area"})
    del carries_pen[0:len(carries_pen)-n]
    # Miscontrolled
    miscontrol = possessionSoup.find_all("td",{"data-stat":"miscontrols"})
    del miscontrol[0:len(miscontrol)-n]
    # Dispossesed
    dispossessed = possessionSoup.find_all("td",{"data-stat":"dispossessed"})
    del dispossessed[0:len(dispossessed)-n]
    # passes received
    received = possessionSoup.find_all("td",{"data-stat":"passes_received"})
    del received[0:len(received)-n]
    # progressive passes received
    prog_received = possessionSoup.find_all("td",{"data-stat":"progressive_passes_received"})
    del prog_received[0:len(prog_received)-n]

    for i in range(n):
        try:
            touches[i] = int(str(touches[i]).split("</td")[0].split('">')[1])
        except ValueError:
            touches[i] = 0
        try:
            touches_def_pen[i] = int(str(touches_def_pen[i]).split("</td")[0].split('">')[1])
        except ValueError:
            touches_def_pen[i] = 0
        try:
            touches_def_3rd[i] = int(str(touches_def_3rd[i]).split("</td")[0].split('">')[1])
        except ValueError:
            touches_def_3rd[i] = 0
        try:
            touches_mid_3rd[i] = int(str(touches_mid_3rd[i]).split("</td")[0].split('">')[1])
        except ValueError:
            touches_mid_3rd[i] = 0
        try:
            touches_att_3rd[i] = int(str(touches_att_3rd[i]).split("</td")[0].split('">')[1])
        except ValueError:
            touches_att_3rd[i] = 0
        try:
            touches_att_pen[i] = int(str(touches_att_pen[i]).split("</td")[0].split('">')[1])
        except ValueError:
            touches_att_pen[i] = 0
        try:
            touches_live_ball[i] = int(str(touches_live_ball[i]).split("</td")[0].split('">')[1])
        except ValueError:
            touches_live_ball[i] = 0
        try:
            take_ons[i] = int(str(take_ons[i]).split("</td")[0].split('">')[1])
        except ValueError:
            take_ons[i] = 0
        try:
            take_ons_won[i] = int(str(take_ons_won[i]).split("</td")[0].split('">')[1])
        except ValueError:
            take_ons_won[i] = 0
        try:
            take_ons_tackled[i] = int(str(take_ons_tackled[i]).split("</td")[0].split('">')[1])
        except ValueError:
            take_ons_tackled[i] = 0
        try:
            carries[i] = int(str(carries[i]).split("</td")[0].split('">')[1])
        except ValueError:
            carries[i] = 0
        try:
            carries_dist[i] = int(str(carries_dist[i]).split("</td")[0].split('">')[1])
        except ValueError:
            carries_dist[i] = 0
        try:
            carries_prog_dist[i] = int(str(carries_prog_dist[i]).split("</td")[0].split('">')[1])
        except ValueError:
            carries_prog_dist[i] = 0
        try:
            carries_prog[i] = int(str(carries_prog[i]).split("</td")[0].split('">')[1])
        except ValueError:
            carries_prog[i] = 0
        try:
            carries_fin_3rd[i] = int(str(carries_fin_3rd[i]).split("</td")[0].split('">')[1])
        except ValueError:
            carries_fin_3rd[i] = 0
        try:
            carries_pen[i] = int(str(carries_pen[i]).split("</td")[0].split('">')[1])
        except ValueError:
            carries_pen[i] = 0
        try:
            miscontrol[i] = int(str(miscontrol[i]).split("</td")[0].split('">')[1])
        except ValueError:
            miscontrol[i] = 0
        try:
            dispossessed[i] = int(str(dispossessed[i]).split("</td")[0].split('">')[1])
        except ValueError:
            dispossessed[i] = 0
        try:
            received[i] = int(str(received[i]).split("</td")[0].split('">')[1])
        except ValueError:
            received[i] = 0
        try:
            prog_received[i] = int(str(prog_received[i]).split("</td")[0].split('">')[1])
        except ValueError:
            prog_received[i] = 0

    # Organize the data
    dataframe = pd.DataFrame({
        "Players": players,
        "Nationalities": nationality,
        "Positions": positions,
        "Teams": teams,
        "Birth year": years,
        "Played Games": games,
        "Started games": start_games,
        "Minutes played": minutes,
        "Goals": goals,
        "Assists": assists,
        "Non-penalty goals": non_penalty,
        "Penalty attempted": penalty_att,
        "Penalty made": penalty_made,
        "Expected goals": xG,
        "Non-Penalty expected goals": npxG,
        "Expected assists": xAG,
        "Progressive carries": prgC,
        "Progressive passes": prgP,
        "Progressive passes received": prgR,
        "Total shots attempted": shots,
        "Shots on target": shots_target,
        "Goal per shots": goal_per_shots,
        "Goal per shots on target": goal_shots_target,
        "Average shot distance": shot_distance,
        "Passes attempted": attempted_passes,
        "Completed passes": completed_passes,
        "Percentage of completed passes": completed_perc,
        "Average passing distance": passing_distance,
        "Average progressive passing distance": prog_pass_distance,
        "Short passes attempted": short_passes_attempted,
        "Short passes completed": short_passes_completed,
        "Medium passes attempted": medium_passes_attempted,
        "Medium passes completed": medium_passes_completed,
        "Long passes attempted": long_passes_attempted,
        "Long passes completed": long_passes_completed,
        "Key passes":key_passes,
        "Passes into the final third": passes_1t,
        "Passes into the penalty area": ppa,
        "Crosses into the penalty area": cpa,
        "Live passes": live_passes,
        "Dead passes": dead_passes,
        "Free kick passes": FK_passes,
        "Through ball passes": through_passes,
        "Switches": switches,
        "Crosses":crosses,
        "Throw-ins taken": throw_in,
        "Shot creating action": sca,
        "Live passes that lead to a shot": sca_live,
        "Dead passes that lead to a shot": sca_dead,
        "Take-on that lead to a shot": sca_to,
        "Shot that lead to another shot": sca_sh,
        "Defensive action that lead to a shot": sca_def,
        "Goal creating action": gca,
        "Live passes that lead to a goal": gca_live,
        "Dead passes that lead to a goal": gca_dead,
        "Take-ons that lead to a goal": gca_to,
        "Shots that lead to a goal": gca_sh,
        "Defensive actions that lead to a goal": gca_def,
        "Tackles":tackles,
        "Tackles won": tackles_won,
        "Tackles in the defensive third": tackles_def_3rd,
        "Tackles in the mid third": tackles_mid_3rd,
        "Tackles in the attacking third": tackles_att_3rd,
        "Driblers tackled": tackles_drib,
        "Dribbles challenged": tackles_challenges,
        "Challenges Lost": tackles_lost,
        "Total blocks": blocks,
        "Shots blocked": block_shots,
        "Passes blocked": block_passes,
        "Interceptions": interceptions,
        "Clearences": clearences,
        "Errors": errors,
    })

    return dataframe


def get_stats(df1k):
    # Get the column of minutes
    minutes = list(df1k.loc[:,'Minutes played'])

    # Get the significant statistics

    # Get position's names
    positions = list(df1k.loc[:,'Positions'])
    n = len(positions)
    # Standard statistics
    goal = list(df1k.loc[:,'Goals'])
    assist = list(df1k.loc[:,'Assists'])
    npg = list(df1k.loc[:,'Non-penalty goals'])
    xG = list(df1k.loc[:,'Expected goals'])
    npxg = list(df1k.loc[:,'Non-Penalty expected goals'])
    xa = list(df1k.loc[:,'Expected assists'])
    # Carries statistics
    pc = list(df1k.loc[:,'Progressive carries'])
    pp = list(df1k.loc[:,'Progressive passes'])
    pr = list(df1k.loc[:,'Progressive passes received'])
    # Shooting statistics
    shots = list(df1k.loc[:,'Total shots attempted'])
    shots_target = list(df1k.loc[:,'Shots on target'])
    goal_shots = list(df1k.loc[:,'Goal per shots'])
    goal_shots_target = list(df1k.loc[:,'Goal per shots on target'])
    shots_distance = list(df1k.loc[:,'Average shot distance'])
    # Passing statistics
    att_passes = list(df1k.loc[:,'Passes attempted'])
    compl_passes = list(df1k.loc[:,'Completed passes'])
    avg_passes_dist = list(df1k.loc[:,'Average passing distance'])
    avg_prog_passes_dist = list(df1k.loc[:,'Average progressive passing distance'])
    short_passes_att = list(df1k.loc[:,'Short passes attempted'])
    short_passes_compl = list(df1k.loc[:,'Short passes completed'])
    med_passes_att = list(df1k.loc[:,'Medium passes attempted'])
    med_passes_compl = list(df1k.loc[:,'Medium passes completed'])
    long_passes_att = list(df1k.loc[:,'Long passes attempted'])
    long_passes_compl = list(df1k.loc[:,'Long passes completed'])
    key_passes = list(df1k.loc[:,'Key passes'])
    fin3_passes = list(df1k.loc[:,'Passes into the final third'])
    pen_passes = list(df1k.loc[:,'Passes into the penalty area'])
    live_passes = list(df1k.loc[:,'Live passes'])
    dead_passes = list(df1k.loc[:,'Dead passes'])
    through_passes = list(df1k.loc[:,'Through ball passes'])
    switches = list(df1k.loc[:,'Switches'])
    # Shots creating stats
    sca = list(df1k.loc[:,'Shot creating action'])
    sca_live = list(df1k.loc[:,'Live passes that lead to a shot'])
    sca_dead = list(df1k.loc[:,'Dead passes that lead to a shot'])
    sca_to = list(df1k.loc[:,'Take-on that lead to a shot'])
    sca_sh = list(df1k.loc[:,'Shot that lead to another shot'])
    sca_def = list(df1k.loc[:,'Defensive action that lead to a shot'])
    # Goal creating stats
    gca = list(df1k.loc[:,'Goal creating action'])
    gca_live = list(df1k.loc[:,'Live passes that lead to a goal'])
    gca_dead = list(df1k.loc[:,'Dead passes that lead to a goal'])
    gca_to = list(df1k.loc[:,'Take-ons that lead to a goal'])
    gca_sh = list(df1k.loc[:,'Shots that lead to a goal'])
    gca_def = list(df1k.loc[:,'Defensive actions that lead to a goal'])
    # Defensive stats
    tackles = list(df1k.loc[:,'Tackles'])
    tackles_won = list(df1k.loc[:,'Tackles won'])
    tackles_d3 = list(df1k.loc[:,'Tackles in the defensive third'])
    tackles_m3 = list(df1k.loc[:,'Tackles in the mid third'])
    tackles_a3 = list(df1k.loc[:,'Tackles in the attacking third'])
    tackles_drib = list(df1k.loc[:,'Driblers tackled'])
    tackles_chal = list(df1k.loc[:,'Dribbles challenged'])
    tackles_lost = list(df1k.loc[:,'Challenges Lost'])
    blocks = list(df1k.loc[:,'Total blocks'])
    blocks_sh = list(df1k.loc[:,'Shots blocked'])
    block_pass = list(df1k.loc[:,'Passes blocked'])
    interception = list(df1k.loc[:,'Interceptions'])
    clearences = list(df1k.loc[:,'Clearences'])
    errors = list(df1k.loc[:,'Errors'])

    # Create players array
    Players_stats = []
    for i in range(n):
        player = []
        m = minutes[i]

        player.append((assist[i]/m, npg[i]/m, npxg[i]/m, xa[i]/m, pc[i]/m))

        player.append((pp[i]/m,pr[i]/m,shots[i]/m,shots_distance[i],att_passes[i]/m))

        player.append((avg_passes_dist[i], short_passes_att[i]/m, med_passes_att[i]/m, long_passes_att[i]/m, key_passes[i]/m))

        player.append((fin3_passes[i]/m,pen_passes[i]/m,sca[i]/m,gca[i]/m,tackles[i]/m))

        player.append((tackles_won[i]/m,blocks[i]/m,interception[i]/m,clearences[i]/m,errors[i]/m))

        Players_stats.append(player)

    return Players_stats, positions

def get_model(lenght):
    
    shape = (lenght, lenght, 1)

    # Create model
    model = tf.keras.models.Sequential([
        # First layer - Input layer
        tf.keras.layers.Dense(200, activation="relu", input_shape=shape),
        # Second layer - Hidden Dense layer
        tf.keras.layers.Dense(200, activation="sigmoid"),
        # Third layer - Hidden Dense layer
        tf.keras.layers.Dense(25, activation="sigmoid"),
        # Fourth layer - Hidden dropout layer
        tf.keras.layers.Dropout(0.1),
        # Fifth layer - Hidden flatten layer
        tf.keras.layers.Flatten(),
        # Sixth layer - Hidden Dense layer
        tf.keras.layers.Dense(128, activation="sigmoid"),
        # Seventh layer - Hidden dropout layer
        tf.keras.layers.Dropout(0.1),
        # Eight layer - Output layer
        tf.keras.layers.Dense(len(CATEGORIES), activation="softmax")
    ])

    # Train the neural network
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def main():

    # Path of the database
    current_dir = os.getcwd()
    parent_dir = Path(current_dir).parent
    dataframe_excel_path = Path(os.path.join(parent_dir, 'text', 'SerieA-stats.xlsx'))

    # If the database doesn't exist create it
    if not Path(dataframe_excel_path).is_file():
        print("Scrapping from web")
        dataframe = scrap()
        dataframe.to_excel(dataframe_excel_path)
    else:
        print("Read from file")
        dataframe = pd.read_excel(dataframe_excel_path)

    # Filter the dataframe for players that has played > 1k minutes
    df1k = dataframe.loc[dataframe['Minutes played'] > 1000]

    # Get the arrays that stores statistics and positions for each player with > 1k minutes
    stats, positions = get_stats(df1k)

    # Convert positions to their numerical value
    labels = []
    for pos in positions:
        pos = pos.split(",")[0]
        if pos in CATEGORIES:
            labels.append(CATEGORIES.index(pos))

    lenght = len(stats[0])

    model = get_model(lenght)

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(stats), np.array(labels) , test_size=TEST_SIZE
        )
    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file
    filename = Path(os.join(parent_dir, 'model'))
    model.save(filename)
    print(f'Model saved to {filename}.')

if __name__=="__main__":
    main()