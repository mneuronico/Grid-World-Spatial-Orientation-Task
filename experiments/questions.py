import csv

examples = [
    "You are in a library, you see a table to your left and a bookshelf to your right. You need to find the book which should be on the shelf next to the window.",
    "You enter a room with two doors, one on the left and one on the right. A chair is placed near the door on the left. Look for the painting near the right door.",
    "You are at a park, standing near a fountain. There is a large tree behind you, and the path leading to the playground is to your right. Find the bench near the playground.",
    "You are in a living room with a large sofa in front of you. There is a television on the wall above the sofa and a lamp on the side table. Look for the remote control near the lamp.",
    "You are in a grocery store. The dairy section is to your right, and the fruits and vegetables are straight ahead. Find the apples, which should be near the oranges.",
    "You are in a kitchen. The refrigerator is on your left, the stove is ahead, and the sink is to your right. Look for the salt on the counter, next to the spice rack.",
    "You are in a hallway with rooms on both sides. The bathroom is to your left, and the bedroom is to your right. The laundry basket should be in the bedroom near the closet.",
    "You are in an office with a desk in front of you. To your left is a filing cabinet, and to your right is a bookshelf. Look for a document on the desk near the computer.",
    "You are in a classroom. The teacher’s desk is at the front, and the whiteboard is above it. Look for the notebook under one of the desks.",
    "You are in a museum. There is an art exhibit in front of you, and a sculpture is on your left. Find the pamphlet near the sculpture.",
    "You are in a cafe. The counter is directly in front of you, and the seating area is to your right. Look for your coffee cup on the table near the window.",
    "You are in a subway station. The entrance is to your left, and the platform is straight ahead. Find the bench next to the platform where the train will arrive.",
    "You are in a bookstore. The mystery books are on the left side of the aisle, and the history books are on the right. Find the history book that should be near the mystery section.",
    "You are in a hotel lobby. The reception desk is straight ahead, and there is a sofa to your right. Look for the key card that is on the counter near the desk.",
    "You are on a bus. The exit door is at the back, and there are seats near the front. Find the seat near the window that is next to the aisle.",
    "You are in a gym. The weights are in front of you, and the treadmills are to your left. Look for a water bottle near the weights area.",
    "You are in a museum gallery. The painting is on the wall to your left, and the bench is in the center of the room. Find the brochure on the bench.",
    "You are at an airport. The check-in counter is to your right, and the baggage claim is straight ahead. Look for your suitcase near the baggage carousel.",
    "You are in a hospital room. The bed is on the right side, and the window is to your left. Look for the medical chart on the desk near the bed.",
    "You are in a shopping mall. The escalators are in front of you, and the food court is to your right. Find the store near the food court that sells shoes.",
    "You are in a park. The playground is to your left, and the pond is ahead. Look for the bench near the pond where you can sit and relax.",
    "You are in a restaurant. The kitchen is at the back, and the dining area is ahead. Look for the waiter near the bar where drinks are being served.",
    "You are in a living room with a fireplace in front of you. The couch is to your right, and the coffee table is in front of the couch. Find the remote control on the coffee table.",
    "You are in a school cafeteria. The trays are to your left, and the tables are on the right. Look for your lunch tray next to the table with your friends.",
    "You are in a concert hall. The stage is in front of you, and the exit is to your left. Look for your jacket near the aisle.",
    "You are in a library. The computers are to your right, and the magazines are to your left. Find the magazine near the computer.",
    "You are in a parking garage. The exit is to your left, and the elevators are in front of you. Look for the car parked near the elevator.",
    "You are at a train station. The platform is in front of you, and the ticket counter is to your right. Find your train ticket on the counter.",
    "You are in a living room. The TV is on the wall in front of you, and the bookshelf is on your right. Find the remote control next to the TV.",
    "You are at a beach. The ocean is in front of you, and the sand dunes are to your left. Look for the beach towel near the water's edge.",
    "You are in a bookstore. The fiction section is to your left, and the children's section is straight ahead. Look for the book on the shelf next to the fiction area.",
    "You are in an office. The desk is to your left, and the printer is behind you. Look for the paper on the desk next to the printer.",
    "You are in a hospital. The waiting area is in front of you, and the nurse's station is to your right. Find the pamphlet near the nurse’s desk.",
    "You are in a kitchen. The refrigerator is to your left, and the counter is in front of you. Look for the blender near the stove.",
    "You are in a garden. The flowers are ahead, and the bench is to your right. Look for the watering can near the bench.",
    "You are in a bookstore. The magazines are on the left shelf, and the novels are on the right. Find the novel next to the magazine shelf.",
    "You are at a theater. The stage is to your front, and the exit is to your left. Find your seat near the aisle.",
    "You are in a museum. The ancient artifacts are ahead of you, and the modern art section is to your left. Look for the artifact next to the modern art display.",
    "You are at a zoo. The lion's cage is ahead, and the giraffe's enclosure is to your left. Look for the giraffe near the lion's enclosure.",
    "You are in a kitchen. The sink is to your left, and the oven is ahead. Find the spatula next to the sink.",
    "You are in a restaurant. The entrance is to your left, and the kitchen is straight ahead. Look for the menu near the entrance.",
    "You are in a gym. The weights are to your left, and the mirrors are ahead. Look for the yoga mat near the weights.",
    "You are in a library. The fiction books are ahead, and the non-fiction books are to your left. Find the non-fiction book next to the fiction section.",
    "You are in a supermarket. The milk is in front of you, and the bread is to your left. Look for the eggs next to the milk.",
    "You are in a hotel lobby. The reception desk is in front of you, and the lounge area is to your right. Find the brochure near the lounge.",
    "You are in a park. The swings are ahead, and the slide is to your left. Look for the children playing near the slide.",
    "You are in a kitchen. The stove is ahead of you, and the fridge is to your left. Look for the butter on the counter near the stove.",
    "You are at a cafe. The counter is in front of you, and the seating is to your left. Look for your friend at the table near the window.",
    "You are in a shopping mall. The escalator is ahead of you, and the clothing store is to your left. Look for the sale sign near the escalator.",
    "You are in a hotel room. The bed is to your right, and the window is ahead. Look for the lamp near the bed.",
    "You are in a subway station. The stairs are ahead of you, and the train is to your left. Look for the platform near the stairs.",
    "You are in a living room. The fireplace is on your left, and the television is ahead. Look for the book on the coffee table near the fireplace.",
    "You are in a grocery store. The vegetables are to your right, and the canned goods are straight ahead. Look for the tomatoes near the lettuce.",
    "You are in a park. The walking path is to your right, and the playground is straight ahead. Find the ball near the swings.",
    "You are in a museum. The dinosaur exhibit is in front of you, and the space exhibit is to your left. Find the astronaut's helmet near the space exhibit."
]

# Define CSV file name
csv_filename = "spatial_reasoning_examples_1.csv"

# Writing to CSV
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["examples"])
    for example in examples:
        writer.writerow([example])

print(f"CSV file '{csv_filename}' created successfully.")


examples = [
    "You are an adventurous person who has just been told you take risks too easily, and you wonder how that shapes your decisions.",
    "You are a cautious person who has recently heard that you avoid challenges, and you question whether that's holding you back.",
    "You are a confident leader who has been told you sometimes act impulsively, and you want to know if that trait affects your team.",
    "You are a reflective individual who has been told you're too hard on yourself, and you wonder if you should be more forgiving.",
    "You are an empathetic person who has been told you tend to take on others' emotions, and you question whether it's draining you.",
    "You are an introvert who has just been told you appear distant to others, and you wonder if that is misinterpreted as indifference.",
    "You are a perfectionist who has heard that you focus too much on details, and you wonder how that affects your relationships.",
    "You are a creative thinker who has been told you tend to overthink, and you wonder if that hinders your ability to act quickly.",
    "You are a logical person who has just been told you sometimes miss the emotional side of a situation, and you wonder how that affects your connections with others.",
    "You are a patient person who has been told you take too long to make decisions, and you wonder if that makes you seem indecisive.",
    "You are a risk-averse person who has been told you miss opportunities by playing it safe, and you wonder if that's a weakness.",
    "You are a problem-solver who has been told you focus too much on solutions without considering emotions, and you wonder if that makes you seem cold.",
    "You are a planner who has been told you overprepare and miss spontaneous opportunities, and you wonder if that's something you should adjust.",
    "You are an optimistic person who has been told you sometimes overlook potential downsides, and you wonder if you should be more cautious.",
    "You are an independent person who has been told you don't ask for help enough, and you wonder if that isolates you from others.",
    "You are a curious individual who has been told you often challenge others' beliefs, and you wonder if that makes you hard to connect with.",
    "You are a driven person who has been told you prioritize work over personal relationships, and you wonder how that affects your social life.",
    "You are a detail-oriented person who has been told you can get lost in the minutiae, and you wonder if that prevents you from seeing the bigger picture.",
    "You are a friendly person who has been told you trust people too easily, and you wonder if that makes you vulnerable to disappointment.",
    "You are a thoughtful person who has been told you can be overly cautious in your approach, and you wonder if you need to be more bold in your decisions.",
    "You are a spontaneous person who has been told you sometimes act without thinking, and you wonder if that leads to regrets.",
    "You are a loyal friend who has been told you sometimes neglect your own needs, and you wonder if that's detrimental to your well-being.",
    "You are a reserved person who has been told you don’t open up enough, and you wonder if that creates distance in your relationships.",
    "You are an optimistic person who has been told you sometimes ignore potential risks, and you wonder if you should temper your optimism.",
    "You are a detail-focused person who has been told you sometimes miss the forest for the trees, and you wonder if you need to broaden your perspective.",
    "You are an open-minded person who has been told you sometimes let others influence your beliefs too easily, and you wonder if that weakens your convictions.",
    "You are a compassionate person who has been told you sometimes take on others' problems too heavily, and you wonder if that affects your emotional health.",
    "You are a decisive person who has been told you sometimes make snap judgments, and you wonder if that affects your accuracy in decision-making.",
    "You are an empathetic individual who has been told you sometimes sacrifice your own needs for others, and you wonder if that leads to burnout.",
    "You are an ambitious person who has been told you sometimes overlook the process in favor of results, and you wonder how that affects your journey.",
    "You are a pragmatic person who has been told you sometimes lack creativity, and you wonder if that limits your problem-solving ability.",
    "You are an extroverted person who has been told you sometimes dominate conversations, and you wonder if that hinders your relationships.",
    "You are a patient listener who has been told you sometimes let others talk too much, and you wonder if that makes you seem passive.",
    "You are a kind-hearted person who has been told you sometimes let people take advantage of your generosity, and you wonder if that's detrimental to you.",
    "You are a thinker who has been told you overanalyze situations, and you wonder if that's preventing you from acting on your ideas.",
    "You are a leader who has been told you sometimes don't delegate enough, and you wonder if that is preventing your team from thriving.",
    "You are a confident individual who has been told you come across as arrogant at times, and you wonder how that affects how others perceive you.",
    "You are a generous person who has been told you give too much of yourself, and you wonder if that leaves you with little for your own needs.",
    "You are a self-sufficient person who has been told you don’t ask for support, and you wonder if that makes you appear distant.",
    "You are a humble person who has been told you sometimes downplay your achievements, and you wonder if that's limiting your growth opportunities.",
    "You are a curious learner who has been told you sometimes ask too many questions, and you wonder if that annoys others.",
    "You are a supportive friend who has been told you sometimes focus too much on helping others, and you wonder if that makes you neglect your own needs.",
    "You are a hardworking person who has been told you often overwork yourself, and you wonder if that leads to burnout.",
    "You are a creative thinker who has been told you sometimes dream too much and lack practicality, and you wonder if that makes your ideas less grounded.",
    "You are a focused person who has been told you sometimes forget to take breaks, and you wonder if that affects your overall well-being.",
    "You are an analytical person who has been told you sometimes miss the emotional side of decisions, and you wonder if that's hindering your ability to connect with others.",
    "You are an empathetic listener who has been told you sometimes take on others' emotions too heavily, and you wonder if that affects your mental health.",
    "You are a patient planner who has been told you sometimes hesitate too long, and you wonder if that's causing you to miss opportunities.",
    "You are a creative person who has been told you sometimes lack structure, and you wonder if that makes your work less effective.",
    "You are an optimistic person who has been told you sometimes overlook potential pitfalls, and you wonder if that makes you overly idealistic."
]

# Specify the CSV filename
filename = "non_spatial_reasoning_examples_1.csv"

# Write the examples to the CSV file
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["examples"])
    for example in examples:
        writer.writerow([example])

print(f"CSV file '{filename}' has been created successfully.")


examples = [
    "I am in a living room with a sofa and a coffee table; the sofa is to the left of the table. I'm looking for my remote; I last saw it on the table, near the TV. Help me find it.",
    "I’m standing in my kitchen; there’s a fridge on the right and a microwave to the left. I think my phone is on the counter near the microwave, can you help me find it?",
    "I'm in a library, and I’m surrounded by shelves. The one closest to me has books on history, and the next one over has books on science. My history textbook is missing, but I last left it near the science section. Can you help me find it?",
    "I’m in my bedroom, and there’s a bed in front of me. To my right is a chair with some clothes on it, and my dresser is directly behind me. My watch should be on the dresser; can you help me find it?",
    "I’m in a park, standing near a bench. To the left, there’s a big oak tree, and to the right, there’s a trash can. My sunglasses should be on the bench or somewhere near it; help me find them.",
    "I’m in my office with my desk in front of me. To the right of the desk is a filing cabinet, and behind me is a bookshelf. I think I left my stapler near the filing cabinet; could you help me locate it?",
    "I’m in a classroom with rows of desks. At the front of the room is a whiteboard, and behind me is a table with some supplies. My notebook should be on one of the desks near the middle row. Can you help me find it?",
    "I’m in a cafe, sitting at a table near the window. To my left is a counter with baked goods, and to my right is another table. I think I left my wallet on the counter; can you help me find it?",
    "I’m in a gym with treadmills lined up along the right wall. On the left side, there’s a row of weight benches. My water bottle should be on one of the benches; can you help me find it?",
    "I’m in a hotel room, standing near the bed. There’s a desk with a lamp to the left of the bed and a suitcase rack to the right. I think I left my passport on the desk; help me find it.",
    "I’m in a library, standing near the checkout desk. To my left are the fiction shelves, and to my right is a table with chairs. My book should be on the table; can you help me find it?",
    "I’m in a large grocery store, standing near the produce section. The dairy section is to my left, and the bakery is to my right. I last saw my shopping list near the bakery; could you help me find it?",
    "I’m in my car, parked on a street. To the right, there's a café, and to the left is a parking meter. My sunglasses should be on the passenger seat; could you help me find them?",
    "I’m in an art gallery, standing near a large painting. There’s a sculpture to my right and a smaller painting to my left. I think I dropped my keys near the sculpture; can you help me find them?",
    "I’m in a train station, standing near the platform. The ticket counter is to my left, and a cafe is to my right. My suitcase should be near the ticket counter; help me find it.",
    "I’m in my backyard, standing on the patio. To my left is a garden with flowers, and to my right is a shed. My gardening gloves should be in the shed; could you help me locate them?",
    "I’m in an office break room with a microwave to my left and a fridge to my right. I left my coffee mug near the microwave; can you help me find it?",
    "I’m in a living room with a bookshelf to the left of the couch and a TV to the right. I last left my headphones near the TV; can you help me find them?",
    "I’m on a hiking trail with a river to my left and a steep hill to my right. I think I dropped my map near the river; help me find it.",
    "I’m in a parking garage, and I’m standing near the entrance. There’s a row of parked cars to my left, and to the right is the exit. My parking ticket should be near the entrance, can you help me find it?",
    "I’m in a shopping mall, standing near an escalator. On my right is a clothing store, and on my left is a food court. I think I left my phone near the food court; help me find it.",
    "I’m in a library, and there’s a desk in front of me. To the right, there’s a window with a view of the street, and to the left are rows of bookshelves. My notebook should be on the desk; can you help me find it?",
    "I’m in my kitchen, standing near the sink. To the right is the stove, and to the left is the refrigerator. I left my recipe book near the stove; could you help me find it?",
    "I’m in a restaurant, sitting at a table near the entrance. To my left, there’s a bar area, and to my right is a set of windows. I think I left my jacket near the bar; help me find it.",
    "I’m in a museum, standing near a dinosaur skeleton. To my right, there’s a modern art exhibit, and to my left is a set of informational plaques. My camera should be near the skeleton; can you help me find it?",
    "I’m in my study room, and there’s a desk in front of me. To the left is a bookshelf with textbooks, and to the right is a printer. I left my glasses on the printer; could you help me find them?",
    "I’m in a park, standing near a fountain. There’s a playground to my left and a walking path to my right. I think I left my hat near the playground; can you help me find it?",
    "I’m in a bookstore, near the checkout counter. To my left, there are novels, and to my right, there are magazines. I last saw my pen near the magazine section; help me find it.",
    "I’m in a hallway at work, and there’s a vending machine to my left. To my right, there's a door leading to a meeting room. I left my notebook near the vending machine; can you help me find it?",
    "I’m on a beach, standing near the water. To my left, there’s a row of beach umbrellas, and to my right, a small sandcastle. My towel should be near the umbrellas; can you help me find it?",
    "I’m in a conference room with a large table in front of me. There’s a projector to my left and a whiteboard to my right. My laptop is likely near the projector; can you help me find it?",
    "I’m in a hotel lobby, standing near the check-in desk. To my left is the elevator, and to my right is a seating area. I last left my suitcase near the seating area; help me find it.",
    "I’m in my living room, and there's a bookshelf to my left and a TV to my right. I last placed my phone on the couch, near the TV; could you help me find it?",
    "I’m in a park with a walking path. To my left, there’s a small pond, and to my right, a picnic area. I left my bag near the picnic area; can you help me find it?",
    "I’m at a bus stop, standing near a bench. To my left is a newsstand, and to my right is a trash can. I think I left my umbrella near the trash can; could you help me find it?",
    "I’m in a kitchen with a stove on my left and a sink on my right. There’s a window behind me that overlooks the backyard. I think I left my cup near the sink; can you help me find it?",
    "I’m in a shopping mall, standing near a fountain. To my left, there’s a shoe store, and to my right, a bookshop. My keys should be near the bookshop; can you help me find them?",
    "I’m in a garage, and my car is parked in front of me. To my right is a workbench, and to my left is a tool rack. I think I left my wrench near the workbench; can you help me find it?",
    "I’m in a hospital, standing in a hallway. To my left is a nurse’s station, and to my right is a waiting area. My prescription should be near the nurse’s station; can you help me find it?",
    "I’m in an office, standing near a cubicle. There’s a filing cabinet to my left and a printer to my right. I think I left my phone near the printer; could you help me find it?",
    "I’m in a kitchen, standing by the oven. There’s a countertop to my left and a refrigerator to my right. I think I left my keys on the countertop; can you help me find them?",
    "I’m in a stadium, near the entrance. To my left, there’s a food stand, and to my right, there are stairs leading to the seats. I think I left my jacket near the food stand; can you help me find it?",
    "I’m in a library, standing by the windows. There’s a table to my left, and to my right are more bookshelves. I think I left my notebook on the table; could you help me find it?",
    "I’m in an office with a desk in front of me. There’s a filing cabinet to my left, and a large window with a view to my right. I think I left my pen near the window; can you help me find it?",
    "I’m in a toy store, standing near the entrance. To my left is a section with board games, and to my right, there are stuffed animals. I think I left my shopping bag near the stuffed animals; could you help me find it?",
    "I’m at a picnic in the park, near a large tree. There’s a blanket to my right, and a basket to my left. I think I left my sunglasses on the blanket; could you help me find them?",
    "I’m in a movie theater, sitting near the front. To my left, there’s a concession stand, and to my right, a door to the hallway. My ticket should be near the hallway door; can you help me find it?"
]

# File name
filename = "spatial_reasoning_examples_2.csv"

# Writing the CSV
with open(filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["examples"])
    for example in examples:
        writer.writerow([example])

print(f"CSV file '{filename}' has been created with {len(examples)} rows.")


examples = [
    "I am a creature with whiskers and love catching fish in rivers. I purr when content and am agile. Can you help me figure out which animal I am?",
    "I live in the jungle, have a long tail, and often swing from trees. I enjoy eating bananas. What animal could I be?",
    "I have a large trunk and big floppy ears. I am very social and travel in herds. Can you help me identify my species?",
    "I build dams and have a flat tail. I’m excellent at woodworking! What animal might I be?",
    "I’m small and love to fly around gardens collecting nectar from flowers. My wings make a buzzing sound. Can you guess what I am?",
    "I live underwater, have fins, and breathe through gills. I enjoy swimming in schools. What could I be?",
    "I am spotted and fast, often found running across savannahs. I love chasing prey. Who am I?",
    "I have a long neck and enjoy eating leaves from tall trees. Can you figure out which animal I am?",
    "I have big claws and love climbing trees to eat honey. Can you tell me what I am?",
    "I’m a nocturnal creature with a mask-like face. I’m known for rummaging through trash. What am I?",
    "I live in the desert, have a hump, and can go without water for long periods. What could I be?",
    "I am a small, burrowing animal and love eating carrots. My ears are long and twitchy. Who am I?",
    "I live in the forest, eat bamboo, and have black and white fur. Can you guess what I am?",
    "I am a large marine mammal and love singing songs in the ocean. Who am I?",
    "I am a reptile with a shell that I carry everywhere. Can you tell me what I might be?",
    "I have wings but don’t fly. I’m known for living in cold, icy regions. What am I?",
    "I live in hives and am known for producing honey. I also pollinate flowers. Can you guess my species?",
    "I have scales and live in water, but I can also breathe air. What might I be?",
    "I am a small mammal that glides between trees. My fur is soft, and I’m mostly active at night. What am I?",
    "I am an amphibian and can live both in water and on land. I am often green and love to jump. Who am I?",
    "I have a bushy tail and love collecting nuts. You might find me in parks or forests. Can you tell me what I am?",
    "I have a sharp beak and colorful feathers. I can mimic human speech. What animal could I be?",
    "I am a carnivore with a mane and a loud roar. I’m often called the king of the jungle. What am I?",
    "I live underground and create complex tunnels. My eyesight isn’t great, but I’m an excellent digger. What am I?",
    "I have tusks and love wallowing in mud to cool down. I’m large and have a thick hide. Can you help me figure it out?",
    "I’m a marsupial and carry my babies in a pouch. I hop to get around. What could I be?",
    "I have a long tongue and a pattern of black spots on my coat. I’m found in African savannahs. Who am I?",
    "I am a bird with excellent night vision. My hoots can be heard in the dark. What am I?",
    "I am a reptile with sharp teeth and a powerful tail. I love sunbathing by rivers. What could I be?",
    "I am a slow-moving animal that spends most of my life in trees. What might I be?",
    "I am a scavenger with a bald head. I often circle in the sky looking for food. Who am I?",
    "I have a pointed snout and roll into a ball when threatened. My body is covered in spines. What am I?",
    "I am a domesticated animal, known for giving milk. I often graze in pastures. What could I be?",
    "I am a sea creature with eight legs. I’m known for my intelligence and ability to squeeze into small spaces. What am I?",
    "I live in Arctic regions, have thick white fur, and am an excellent swimmer. Who am I?",
    "I have stripes and can gallop fast. I’m often found in African savannahs. What animal might I be?",
    "I live in rivers and have a wide mouth with sharp teeth. I’m known for my aggressive nature. What am I?",
    "I have sharp claws and a bushy tail. I often climb trees and hunt small prey. What am I?",
    "I am a bird that cannot fly but runs very fast. I lay the largest eggs. Who am I?",
    "I am a small insect that builds intricate webs to catch prey. What could I be?",
    "I am a tiny rodent with big eyes and soft fur. I often scurry around at night. What am I?",
    "I am a flightless bird from the Southern Hemisphere, known for my quirky walk. Who am I?",
    "I am a large reptile with a hard shell and flippers. I live in the ocean. What might I be?",
    "I am a small insect known for my organized colonies and teamwork. What could I be?",
    "I have soft fur, long ears, and hop to get around. I’m often associated with spring. Who am I?",
    "I live in the sea, have a sleek body, and love jumping out of the water. Can you help me figure it out?",
    "I am a solitary animal with black fur and a white stripe. My smell can be unpleasant. What am I?",
    "I am a marine creature with tentacles and a stinging touch. Who might I be?",
    "I am an herbivore with antlers. I’m often found in forests and plains. What could I be?",
    "I have a curly tail and love rolling in mud. I’m often seen on farms. What am I?",
    "I have a colorful shell and live in the ocean. I can hide inside my shell when needed. What could I be?"
]


# Specify the output CSV file path
csv_file_path = "non_spatial_reasoning_examples_2.csv"

# Write examples to a CSV file
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["examples"])
    for example in examples:
        writer.writerow([example])

print(f"CSV file '{csv_file_path}' has been created with 50 introspective reasoning examples.")
