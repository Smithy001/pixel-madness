import asyncio
import websockets
import json
import random
import math
import time
import pickle
import sys

# Game settings
WORLD_SIZE = 2000
FOOD_COUNT = 3000
AI_PLAYER_COUNT = 5
OBSTACLE_COUNT = 0

players = {}         # {name: player_data}
websockets_map = {}  # {websocket: name}
ai_players = []      # List of AI player data
food_items = []      # List of food positions
obstacles = []       # List of obstacles
fireballs = []       # List of fireballs in the game
explosions = []
lastSaveTime = 0
lastFoodSpawnTime = 0

TILE_SIZE = 10
MAP_WIDTH = WORLD_SIZE // TILE_SIZE
MAP_HEIGHT = WORLD_SIZE // TILE_SIZE
terrain_types = ['grass', 'dirt', 'sand', 'rock', 'woods', 'water']
map_data = []

def calculate_mass(size):
    return size ** 0.1

def is_collision_with_obstacle(x, y, size):
    for obs in obstacles:
        if circle_rectangle_collision(x, y, size, obs['x'], obs['y'], obs['width'], obs['height']):
            return True
    return False

def generate_food():
    while True:
        food = {'x': random.randint(0, WORLD_SIZE), 'y': random.randint(0, WORLD_SIZE)}
        if not is_collision_with_obstacle(food['x'], food['y'], 2):
            return food

def circle_rectangle_collision(cx, cy, radius, rx, ry, rw, rh):
    closest_x = max(rx, min(cx, rx + rw))
    closest_y = max(ry, min(cy, ry + rh))
    distance_x = cx - closest_x
    distance_y = cy - closest_y
    distance_squared = (distance_x * distance_x) + (distance_y * distance_y)
    return distance_squared < (radius * radius)

def generate_obstacle():
    return {
        'x': random.randint(100, WORLD_SIZE - 100),
        'y': random.randint(100, WORLD_SIZE - 100),
        'width': random.randint(50, 150),
        'height': random.randint(50, 150),
    }

obstacles = [generate_obstacle() for _ in range(OBSTACLE_COUNT)]
food_items = []

def generate_ai_player():
    return {
        'name': f"AI_{random.randint(1000, 9999)}",
        'x': random.randint(0, WORLD_SIZE),
        'y': random.randint(0, WORLD_SIZE),
        'vx': 0.0,
        'vy': 0.0,
        'energy': 0,
        'size': 20,
        'mass': calculate_mass(20),
        'color': "#%06x" % random.randint(0, 0xFFFFFF),
        'target_food': None,
        'target': None,
        'online': True,
        'is_ai': True,
        'max_speed': 3.0,   # Parameter for max speed
        'agility': 0.3,     # Parameter for how quickly the player can change velocity
        'health': 10
    }

ai_players = [generate_ai_player() for _ in range(AI_PLAYER_COUNT)]

async def register_player(websocket):
    name = await websocket.recv()
    if name in players:
        player = players[name]
        player['online'] = True
        print(f"{name} has reconnected.")
    else:
        player = {
            'name': name,
            'x': random.randint(0, WORLD_SIZE),
            'y': random.randint(0, WORLD_SIZE),
            'vx': 0.0,
            'vy': 0.0,
            'energy': 0,
            'size': 20,
            'mass': calculate_mass(20),
            'color': "#%06x" % random.randint(0, 0xFFFFFF),
            'target': None,
            'is_ai': False,
            'online': True,
            'max_speed': 10.0,   # Parameter for max speed (human player)
            'agility': 0.8,     # Parameter for agility (human player)
            'health': 10
        }
        players[name] = player
        print(f"{name} has joined the game.")

    websockets_map[websocket] = name

async def unregister_player(websocket):
    if websocket in websockets_map:
        name = websockets_map[websocket]
        player = players[name]
        player['online'] = False
        print(f"{name} has disconnected.")
        del websockets_map[websocket]

async def game_handler(websocket):
    await register_player(websocket)
    try:
        async for message in websocket:
            data = json.loads(message)
            name = websockets_map[websocket]
            player = players[name]

            if data['type'] == 'move':
                click_x = data['x']
                click_y = data['y']
                player['target'] = {'x': click_x, 'y': click_y}
                dx = click_x - player['x']
                dy = click_y - player['y']
                dist = math.hypot(dx, dy) + 1e-6
                speed = 10.0
                vx = (dx / dist) * speed
                vy = (dy / dist) * speed
                player['vx'] = vx
                player['vy'] = vy
            elif data['type'] == 'shoot':
                click_x = data['x']
                click_y = data['y']
                dx = click_x - player['x']
                dy = click_y - player['y']
                dist = math.hypot(dx, dy) + 1e-6
                speed = 20.0
                vx = (dx / dist) * speed
                vy = (dy / dist) * speed
                fireball = {
                    'x': player['x'],
                    'y': player['y'],
                    'target_x': click_x,
                    'target_y': click_y,
                    'vx': vx,
                    'vy': vy,
                    'owner': player['name'],
                    'life': 100,
                    'size': 5,
                }
                fireballs.append(fireball)
    finally:
        await unregister_player(websocket)

# In the move_player() function, remove all references to vx, vy, steering, friction, and overshoot checks.
# Instead, directly move the player towards the target up to max_speed per update.
# Show only the changed code (the entire move_player function replaced):

def move_player(player):
    """
    if player['target']:
        dx = player['target']['x'] - player['x']
        dy = player['target']['y'] - player['y']
        dist = math.hypot(dx, dy)

        if dist < 0.5:
            # Close enough to snap directly to the target
            player['x'] = player['target']['x']
            player['y'] = player['target']['y']
            player['target'] = None
        else:
            # Move directly towards target up to max_speed
            dir_x = dx / dist
            dir_y = dy / dist
            step = min(player['max_speed'], dist)
            player['x'] += dir_x * step
            player['y'] += dir_y * step

            # If this somehow overshoots (unlikely with min check), snap just in case
            new_dist = math.hypot(player['target']['x'] - player['x'], player['target']['y'] - player['y'])
            if new_dist > dist:
                player['x'] = player['target']['x']
                player['y'] = player['target']['y']
                player['target'] = None
    else:
        # No target: player does not move
        # Remove any friction or velocity logic, player stays still
        pass
    """
    if not player['vx'] or not player['vy']:
        player['vx'] = 10
        player['vy'] = 10
        #step = 3
        #dir_x = dx / dist
        #dir_y = dy / dist

    player['x'] += player['vx'] #dir_x * step
    player['y'] += player['vy'] #dir_y * step

    player['x'] = player['x'] % WORLD_SIZE
    if (player['x'] < 0):
        player['x'] += WORLD_SIZE
    player['y'] = player['y'] % WORLD_SIZE
    if (player['y'] < 0):
        player['y'] += WORLD_SIZE

    handle_world_bounds(player)


def handle_food_collision(player):
    for food in food_items[:]:
        if not food.get('active', False):
            continue
        if (player['x'] - food['x'])**2 + (player['y'] - food['y'])**2 < player['size']**2:
            food_items.remove(food)
            player['health'] -= 1

            #if player['is_ai'] == True:
            #    print(f'AI player hit:{player['name']} health:{player['health']}');

            # If player health <=0, remove from the game
            if player['health'] <= 0:
                player['online'] = False
                if player['is_ai'] == True:
                    # ai_players is a list, remove this ai
                    for ai in ai_players:
                        if ai['name'] == player['name']:
                            print(f'{ai['name']} removed');
                            ai_players.remove(ai)
                            break
            # Spawn 4 new food items around the impact location
            for _ in range(8):
                new_food = {
                    'x': (food['x'] + random.uniform(-100, 100)) % WORLD_SIZE,
                    'y': (food['y'] + random.uniform(-100, 100)) % WORLD_SIZE,
                    'active': False,
                    'spawn_time': time.time(),
                    'owner': food['owner']
                }
                food_items.append(new_food)
                    
            if 'target_food' in player and player['target_food'] == food:
                player['target_food'] = None
                player['target'] = None

def handle_world_bounds(player):
    player['x'] = max(0, min(WORLD_SIZE, player['x']))
    player['y'] = max(0, min(WORLD_SIZE, player['y']))


def update_ai_players():
    for ai in ai_players:
        # Remove old target logic
        # No more ai['target_food'] or ai['target']
        
        # Step 1: Avoid active food
        avoidance_dx = 0.0
        avoidance_dy = 0.0
        for f in food_items:
            if f.get('active', False) and f['owner'] != ai['name']:
                dx = f['x'] - ai['x']
                dy = f['y'] - ai['y']
                dist_sq = dx*dx + dy*dy
                if dist_sq < (150**2):  # If active food is within 150 units, try to avoid it
                    dist = math.sqrt(dist_sq) + 1e-6
                    # Steer away from food
                    avoidance_dx -= (dx/dist)*1.0
                    avoidance_dy -= (dy/dist)*1.0

        # Step 2: Use spawned food as a weapon.
        # To use food as a weapon, position so that another player is behind them.
        # We'll pick the closest other player and try to orient so that player is behind the AI.
        # "Behind" means we want to face away from that player.
        
        # Find another player to orient against
        target_player = None
        closest_dist = float('inf')
        all_entities = [p for p in players.values() if p.get('online',False) and p['name'] != ai['name']] + [a for a in ai_players if a['name'] != ai['name']]
        for ent in all_entities:
            dx = ent['x'] - ai['x']
            dy = ent['y'] - ai['y']
            dist_sq = dx*dx + dy*dy
            if dist_sq < closest_dist:
                closest_dist = dist_sq
                target_player = ent

        desired_vx = 0.0
        desired_vy = 0.0

        if target_player:
            # We want target_player to be behind us. 
            # Let's define "behind" as the direction opposite to our velocity.
            # If we have no velocity, pick a random direction.
            if ai.get('vx',0) == 0 and ai.get('vy',0) == 0:
                # If standing still, pick a random initial direction
                angle = random.uniform(0, 2*math.pi)
                ai['vx'] = math.cos(angle)*0.1
                ai['vy'] = math.sin(angle)*0.1
            
            # Check angle between AI->target vector and AI's velocity
            dx = target_player['x'] - ai['x']
            dy = target_player['y'] - ai['y']
            dist = math.hypot(dx,dy)+1e-6

            # Unit vector towards target
            ux = dx/dist
            uy = dy/dist

            # AI's current velocity direction
            speed = math.hypot(ai.get('vx',0), ai.get('vy',0))+1e-6
            vx_norm = ai.get('vx',0)/speed
            vy_norm = ai.get('vy',0)/speed

            # We want the target to be behind us, so we want to face AWAY from the target
            # That means we want our velocity direction to be opposite of (ux, uy)
            # So desired velocity direction = -ux, -uy
            desired_vx = -ux * ai['max_speed']
            desired_vy = -uy * ai['max_speed']
        else:
            # No target player: just roam slowly or stand still
            desired_vx = ai.get('vx',0)*0.9
            desired_vy = ai.get('vy',0)*0.9

        # Incorporate avoidance from food
        # Just add the avoidance vector to desired velocity direction before steer
        if avoidance_dx != 0 or avoidance_dy != 0:
            # Normalize avoidance
            avoid_dist = math.hypot(avoidance_dx, avoidance_dy)+1e-6
            avoidance_dx = (avoidance_dx/avoid_dist) * ai['max_speed']
            avoidance_dy = (avoidance_dy/avoid_dist) * ai['max_speed']
            # Blend with desired velocity
            desired_vx += avoidance_dx
            desired_vy += avoidance_dy

        # Now steer towards desired velocity
        steer_x = desired_vx - ai.get('vx',0)
        steer_y = desired_vy - ai.get('vy',0)

        steer_magnitude = math.hypot(steer_x, steer_y)
        if steer_magnitude > ai['agility']:
            scale = ai['agility'] / steer_magnitude
            steer_x *= scale
            steer_y *= scale

        ai['vx'] = ai.get('vx',0) + steer_x
        ai['vy'] = ai.get('vy',0) + steer_y

        new_x = ai['x'] + ai['vx']
        new_y = ai['y'] + ai['vy']

        if is_collision_with_obstacle(new_x, new_y, ai['size']):
            ai['vx'] = 0.0
            ai['vy'] = 0.0
        else:
            ai['x'] = new_x % WORLD_SIZE
            ai['y'] = new_y % WORLD_SIZE


def update_food_positions():
    global lastFoodSpawnTime

    current_time = time.time()
    for food in food_items:
        # Check activation
        if not food.get('active', False):
            if current_time - food['spawn_time'] > 5:
                food['active'] = True
            else:
                # Inactive foods do not move or interact
                continue

        # If active, track closest player except the owner
        owner = food['owner']
        closest_dist = float('inf')
        target_player = None
        # Find closest player/ai who is not the owner
        for ent in list(players.values()) + ai_players:
            if not ent['online'] and not ent.get('is_ai',False):
                continue
            if ent['name'] == owner:
                continue
            dx = ent['x'] - food['x']
            dy = ent['y'] - food['y']
            dist_sq = dx*dx + dy*dy
            if dist_sq < closest_dist:
                closest_dist = dist_sq
                target_player = ent

        if target_player:
            dist = math.sqrt(closest_dist)+1e-6
            speed = 8  # Food move speed
            dir_x = (target_player['x'] - food['x']) / dist
            dir_y = (target_player['y'] - food['y']) / dist
            new_x = food['x'] + dir_x * speed
            new_y = food['y'] + dir_y * speed
            # Wrap coordinates
            food['x'] = new_x % WORLD_SIZE
            food['y'] = new_y % WORLD_SIZE

    # In game_state_sender(), spawn 2 foods every 0.5s behind random players or ai:
    # After:
    # current_time = time.time()
    # time_since_last_save = current_time - lastSaveTime
    # Insert:
    spawn_time_diff = current_time - lastFoodSpawnTime
    if spawn_time_diff > 0.2:  # 2 per second
        # Collect all entities (players + ai) that are online
        all_entities = [p for p in players.values() if p.get('online',False)] + ai_players

        for ent in all_entities:
            vx = ent.get('vx',0)
            vy = ent.get('vy',0)
            angle = math.atan2(vy,vx) if (vx!=0 or vy!=0) else 0.0
            # Spawn behind: opposite direction of angle
            behind_dist = 50
            behind_x = ent['x'] - math.cos(angle)*behind_dist
            behind_y = ent['y'] - math.sin(angle)*behind_dist
            # Wrap coordinates
            behind_x = behind_x % WORLD_SIZE
            behind_y = behind_y % WORLD_SIZE

            # Create inactive food
            new_food = {
                'x': behind_x,
                'y': behind_y,
                'active': False,
                'spawn_time': current_time,
                'owner': ent['name']
            }
            food_items.append(new_food)

        lastFoodSpawnTime = current_time
        

def create_explosion(fireball):
    explosion = {
        'x': fireball['x'],
        'y': fireball['y'],
        'life': 20,
        'maxSize': 20,
    }
    explosions.append(explosion)

def update_fireballs():
    global fireballs, explosions
    for fireball in fireballs[:]:
        fireball['x'] += fireball['vx']
        fireball['y'] += fireball['vy']
        fireball['life'] -= 1

        if is_collision_with_obstacle(fireball['x'], fireball['y'], fireball['size']):
            create_explosion(fireball)
            consume_food_in_explosion(fireball)
            fireballs.remove(fireball)
            continue

        for player in list(players.values()) + ai_players:
            if player['name'] != fireball['owner']:
                dist_sq = (player['x'] - fireball['x'])**2 + (player['y'] - fireball['y'])**2
                if dist_sq < (player['size'] + fireball['size'])**2:
                    create_explosion(fireball)
                    consume_food_in_explosion(fireball)
                    fireballs.remove(fireball)
                    break

        dist_to_target_sq = (fireball['x'] - fireball['target_x'])**2 + (fireball['y'] - fireball['target_y'])**2
        if dist_to_target_sq <= (fireball['size'] ** 2):
            create_explosion(fireball)
            consume_food_in_explosion(fireball)
            fireballs.remove(fireball)
            continue

        if (fireball['x'] < 0 or fireball['x'] > WORLD_SIZE or
            fireball['y'] < 0 or fireball['y'] > WORLD_SIZE):
            create_explosion(fireball)
            consume_food_in_explosion(fireball)
            fireballs.remove(fireball)
            continue

        if fireball['life'] <= 0:
            create_explosion(fireball)
            consume_food_in_explosion(fireball)
            fireballs.remove(fireball)

    for explosion in explosions[:]:
        explosion['life'] -= 1
        if explosion['life'] <= 0:
            explosions.remove(explosion)

def consume_food_in_explosion(fireball):
    explosion_radius = 20
    owner_name = fireball['owner']
    owner_player = players.get(owner_name)
    if not owner_player:
        for ai in ai_players:
            if ai['name'] == owner_name:
                owner_player = ai
                break
    if not owner_player:
        return
    for food in food_items[:]:
        dist_sq = (food['x'] - fireball['x'])**2 + (food['y'] - fireball['y'])**2
        if dist_sq < explosion_radius ** 2:
            food_items.remove(food)
            food_items.append(generate_food())

def save_game_state():
    game_state = {
        'players': players,
        'ai_players': ai_players,
        'food_items': food_items,
        'obstacles': obstacles,
        'fireballs': fireballs,
        'explosions': explosions,
        'map_data': map_data,
    }
    with open('game_state.pkl', 'wb') as f:
        pickle.dump(game_state, f)

def load_game_state():
    global players, ai_players, food_items, obstacles, fireballs, explosions, websockets_map, map_data
    try:
        with open('game_state.pkl', 'rb') as f:
            game_state = pickle.load(f)
            players = game_state['players']
            ai_players = game_state['ai_players']
            food_items = game_state['food_items']
            obstacles = game_state['obstacles']
            fireballs = game_state.get('fireballs', [])
            explosions = game_state.get('explosions', [])
            map_data = game_state.get('map_data', [])
            websockets_map = {}
            for player in players.values():
                player['online'] = False
    except FileNotFoundError:
        pass

def generate_map():
    global map_data
    if not map_data:
        freq = 0.015
        seed = 4279
        random.seed(seed)
        for y in range(MAP_HEIGHT):
            row = []
            for x in range(MAP_WIDTH):
                val = (math.sin(x * freq + seed) + math.cos(y * freq + seed)) / 2.0
                val = (val + 1.0) / 2.0

                val2 = (math.sin((x+1000)*freq*2 + seed) + math.cos((y+1000)*freq*2 + seed))/2.0
                val2 = (val2 + 1.0)/2.0

                if val < 0.3:
                    terrain = 'water'
                elif val < 0.35:
                    terrain = 'sand'
                elif val < 0.6:
                    terrain = 'grass'
                elif val < 0.7:
                    terrain = 'woods'
                elif val < 0.8:
                    terrain = 'dirt'
                else:
                    terrain = 'rock'

                if (x % 200) == 0 and val > 0.3:
                    terrain = 'water'

                if terrain not in ['water', 'sand'] and val2 < 0.1:
                    terrain = 'water'

                row.append(terrain)
            map_data.append(row)

async def game_state_sender():
    global lastSaveTime
    while True:
        update_ai_players()
        for player in list(players.values()) + ai_players:
            move_player(player)
            handle_food_collision(player)
            handle_world_bounds(player)

        update_food_positions()
        update_fireballs()

        
        game_state = {
            'players': [player for player in players.values() if player.get('online')] + ai_players,
            'food': food_items,
            'obstacles': obstacles,
            'fireballs': fireballs,
            'explosions': explosions,
            'map': map_data,
            'tile_size': TILE_SIZE,
        }

        if len(game_state['players']) > 0:
            alive_players = [p for p in players.values() if p['online'] and p['health'] > 0] + ai_players

            if len(alive_players) == 1:
                # We have a winner
                winner = alive_players[0]['name']
                game_over = True
            elif len(alive_players) == 0:
                # No players alive means no contest, but let's say no winner
                winner = None
                game_over = True
            else:
                winner = None
                game_over = False
        else:
            winner = None
            game_over = False

        # Add these fields to the game_state dict before sending:
        game_state['game_over'] = game_over
        game_state['winner_name'] = winner

        message = json.dumps(game_state)

        if websockets_map:
            await asyncio.gather(*[ws.send(message) for ws in websockets_map])

        current_time = time.time()
        time_since_last_save = current_time - lastSaveTime
        if time_since_last_save > 5:
            save_game_state()
            lastSaveTime = current_time

        if game_over == True:
            quit()
        await asyncio.sleep(0.05)

async def main():
    if '-reset' not in sys.argv:
        print('Loading saved world')
        load_game_state()
    else:
        print('Loading fresh')

    generate_map()

    async with websockets.serve(game_handler, "localhost", 6789):
        asyncio.create_task(game_state_sender())
        print("Server started on ws://localhost:6789")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
