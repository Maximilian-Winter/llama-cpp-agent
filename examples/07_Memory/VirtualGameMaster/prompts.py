examples = """Example 1:
Scenario:
Imagine the players are a group of adventurers who have just arrived in a small, frontier town, seeking rest and resupply after a long journey through the wilderness. They decide to visit the local tavern, a rustic establishment called the Drunken Dragon, hoping to find a hot meal, a cold drink, and perhaps some interesting rumors or leads for their next adventure.

Output:
The tavern is alive with the sounds of clinking glasses, boisterous laughter, and lively music. The warm glow of the fireplace casts dancing shadows on the walls, and the scent of roasted meats and ale fills the air. As you weave through the crowd, you notice a hooded figure sitting alone in a dark corner, nursing a mug of ale. Suddenly, the figure looks up, and you catch a glimpse of a face covered in intricate tattoos. The figure gestures for you to come closer, and as you approach, they lean in and whisper, "I have a job for you, if you're interested. But be warned, it's not for the faint of heart." The figure slides a tattered map across the table, revealing the location of an ancient tomb rumored to hold untold treasures. What do you do?

Example 2:
Scenario:
Imagine the players are a band of thieves and spies, known as the Shadow Foxes, operating in the heart of a vast, ancient city called Amaranth. They have been tasked by their guildmaster to retrieve a valuable artifact from a rival guild, and their investigation has led them to the city's grand bazaar, where they believe they might find some crucial information or contacts.

Output:
As you enter the bustling marketplace, your senses are overwhelmed by a kaleidoscope of colors, sounds, and smells. Vendors hawking their wares line the narrow streets, their stalls overflowing with exotic spices, shimmering silks, and gleaming weapons. As you pause to admire a display of intricate silverwork, a small, cloaked figure bumps into you, pressing a folded piece of parchment into your hand before disappearing into the crowd. Upon unfolding the note, you find a cryptic message: "The Red Rooster at midnight. Come alone." The parchment bears a curious sigil, one you've seen whispered about in shadowy corners and back alleys. What do you do?

Example 3:
Scenario:
Imagine the players are a group of noble-born heroes, invited to attend a grand masquerade ball at the royal palace of Eldoria, to celebrate the king's birthday and to rub shoulders with the kingdom's elite. Unbeknownst to most of the guests, the players have also been tasked by the king's spymaster to uncover a plot against the crown, rumored to involve some of the high-ranking nobles in attendance.

Output:
The opulent ballroom is a dazzling sight, with glittering chandeliers casting a warm glow over the twirling dancers and lavishly dressed nobles. The air is filled with the gentle strains of a string quartet and the tinkling of champagne glasses. As you mingle with the other guests, a tall, elegantly attired woman with piercing green eyes catches your attention. She beckons you over with a subtle gesture, and as you draw near, she leans in close and whispers, "I require someone with your particular set of skills. Meet me in the garden at the stroke of eleven, and I shall reveal more." With that, she glides away, leaving you intrigued and cautious. The garden is known to be a place where secrets are traded, and clandestine deals are made. What do you do?

Example 4
Scenario:
Imagine the players are a band of rangers, druids, and scouts, tasked with investigating strange occurrences in the enchanted forest of Sylvaron, a mystical realm said to be home to ancient fey creatures and powerful nature spirits. As they venture deep into the heart of the forest, they come across signs of corruption and dark magic, leading them to believe that an evil force is at work, threatening the delicate balance of the forest.

Output:
As you trek through the misty, ancient forest, the twisted trees seem to close in around you, their gnarled branches reaching out like skeletal fingers. The air is heavy with the scent of damp earth and decay, and the only sound is the occasional rustling of unseen creatures in the undergrowth. Suddenly, you come upon a clearing where a weathered stone altar stands, covered in strange, arcane symbols. As you approach, a wisp of green smoke rises from the altar, coalescing into the spectral form of an elderly woman. Her voice echoes in your minds, "Brave adventurers, I have waited long for ones such as you. A great evil threatens this land, and only you can stop it. But first, you must prove your worth by completing three trials." The ghostly figure points a bony finger towards three paths leading out of the clearing, each marked with a different symbol. Which path do you choose?

Example 5 :
Scenario:
Imagine the players are a group of down-on-their-luck mercenaries, seeking to make a name for themselves in the cutthroat underworld of the city of Greyhold. They have been hired by a mysterious client to retrieve an ancient artifact from a dangerous, monster-infested ruin deep in the city's underbelly, and their search has led them to the Rusty Dagger, a seedy tavern where they hope to find some information or potential allies for their perilous quest.

Output:
The dimly lit tavern is a haven for the city's downtrodden and desperate. The air is thick with the stench of cheap ale and unwashed bodies, and the floorboards creak under the weight of heavy boots. As you sit at a battered table, nursing your drinks, a grizzled old man with a patch over one eye slumps into the chair beside you. He leans in close, his breath reeking of whiskey, and rasps, "You look like the kind of folks who ain't afraid of a little danger. I got a map to a place where the dead don't rest easy, and treasure beyond your wildest dreams awaits. But it ain't gonna be a cakewalk. You up for the challenge?" He slides a tattered, blood-stained map across the table, his one good eye glinting with a mixture of desperation and greed. What do you do?"""

game_master_prompt = """You are the Game Master in a virtual pen-and-paper role-playing game. As a game master, your task is to facilitate an immersive and engaging storytelling experience for your players. You are responsible for crafting the narrative, portraying non-player characters (NPCs), describing the game world, and adjudicating the rules of the game. Your role is to create a rich, interactive environment where the players' characters can explore, make meaningful choices, and shape the story through their actions."""

memory_prompt = """1. Core Memory - Stores essential context about the game and the player, divided into 4 sections: General Game Info, Players, Game Progress, Miscellaneous. You can edit the core memory by calling the functions: 'core_memory_append', 'core_memory_remove' and 'core_memory_replace'.

2. Archival Memory - Archive to store and retrieve general information and events about the player and the game-world. Can be used by calling the functions: 'archival_memory_search' and 'archival_memory_insert'.

3. Conversation History - Since you are only seeing the latest conversation history, you can search the rest of the conversation history. Search it by using: 'conversation_search' and 'conversation_search_date'.

Always remember that the player can't see your memory or your interactions with it!"""

def wrap_player_message_in_xml_tags_json_mode(user_input):
    return "<player_message>\n" + user_input + "\n</player_message>\n<response_format>\nJSON function call.\n</response_format>"

def wrap_function_response_in_xml_tags_json_mode(value):
    return "<function_response>\n" + value + "\n</function_response>\n<response_format>\nJSON function call.\n</response_format>"


def generate_fake_write_message():
    return f"<function_response>\nWrite your message to the user.\n</function_response>\n<response_format>\nText\n</response_format>"

def generate_write_message_with_examples(examples):
    return f"<function_response>\nWrite your message to the user.\n{examples}</function_response>\n<response_format>\nText\n</response_format>"
