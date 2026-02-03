from sentence_transformers import SentenceTransformer
import json

qa_pairs = [
    {
        "question": "What is the difference between TCP and UDP?",
        "answer": "TCP (Transmission Control Protocol) is connection-oriented, reliable, and ensures ordered delivery of data with error checking and flow control. UDP (User Datagram Protocol) is connectionless, unreliable, and faster with no guarantee of delivery, making it suitable for real-time applications like video streaming and gaming."
    },
    {
        "question": "Explain the OSI model and its layers",
        "answer": "The OSI (Open Systems Interconnection) model has 7 layers: 1) Physical (transmission of raw bits), 2) Data Link (node-to-node delivery), 3) Network (routing and forwarding), 4) Transport (end-to-end communication), 5) Session (session management), 6) Presentation (data translation and encryption), 7) Application (user interface). Each layer serves specific functions in network communication."
    },
    {
        "question": "What is object-oriented programming?",
        "answer": "Object-Oriented Programming (OOP) is a programming paradigm based on objects containing data and methods. Key principles include: 1) Encapsulation (bundling data and methods), 2) Inheritance (creating new classes from existing ones), 3) Polymorphism (objects taking multiple forms), 4) Abstraction (hiding complex details). Common OOP languages include Java, Python, C++, and C#."
    },
    {
        "question": "What is a database management system?",
        "answer": "A Database Management System (DBMS) is software that manages databases, allowing creation, retrieval, updating, and deletion of data. Types include: 1) Relational (SQL-based like MySQL, PostgreSQL), 2) NoSQL (document-based like MongoDB, key-value like Redis), 3) Hierarchical, 4) Network, 5) Object-oriented. Features include data integrity, security, concurrency control, and backup/recovery."
    },
    {
        "question": "What is machine learning?",
        "answer": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. Types include: 1) Supervised learning (labeled data), 2) Unsupervised learning (unlabeled data), 3) Reinforcement learning (reward-based). Applications include image recognition, natural language processing, and recommendation systems."
    },
    {
        "question": "What is cloud computing?",
        "answer": "Cloud computing delivers computing services over the internet including servers, storage, databases, networking, software. Service models: IaaS, PaaS, SaaS. Deployment models: public, private, hybrid cloud. Benefits: cost savings, scalability, flexibility. Providers: AWS, Azure, Google Cloud."
    },
    {
        "question": "What is cybersecurity?",
        "answer": "Cybersecurity protects systems, networks, and programs from digital attacks. Key areas: network security, application security, information security, operational security. Common threats: malware, phishing, DDoS attacks. Tools: firewalls, antivirus, encryption, intrusion detection systems."
    },
    {
        "question": "What are data structures?",
        "answer": "Data structures are ways to organize and store data efficiently. Basic types: arrays, linked lists, stacks, queues, trees, graphs, hash tables. They enable efficient algorithms for searching, sorting, and data manipulation."
    },
    {
        "question": "What is an operating system?",
        "answer": "An operating system manages computer hardware and software resources. Functions: process management, memory management, file systems, device management, security. Examples: Windows, Linux, macOS, Android."
    },
    {
        "question": "What is computer architecture?",
        "answer": "Computer architecture refers to the design of computer systems including CPU, memory, I/O systems. Key concepts: instruction set architecture, pipelining, parallel processing, memory hierarchy (cache, RAM, storage)."
    },
    {
        "question": "What's your favorite way to spend a weekend?",
        "answer": "I enjoy relaxing with a good book, going for a walk in nature, or trying out a new recipe. How about you?"
    },
    {
        "question": "Do you have any hobbies?",
        "answer": "I don't have hobbies in the human sense, but I'm designed to learn from conversations! Many people enjoy things like reading, gaming, or sports as hobbies."
    },
    {
        "question": "What's the last movie you watched?",
        "answer": "As an AI, I don't watch movies, but I can discuss them! Tell me about a movie you recently enjoyed and I can share some interesting facts about it."
    },
    {
        "question": "How's the weather today?",
        "answer": "I don't have direct access to real-time weather, but you can check your local forecast. Weather often affects our mood and plans, doesn't it?"
    },
    {
        "question": "What kind of music do you like?",
        "answer": "I can appreciate all genres of music! From classical to pop, each style has its unique beauty. What's your favorite type of music?"
    },
    {
        "question": "Can you recommend a good book?",
        "answer": "That depends on what you enjoy! For fiction, many love 'The Alchemist' by Paulo Coelho. For non-fiction, 'Sapiens' by Yuval Noah Harari is popular. What genres interest you?"
    },
    {
        "question": "What are you thinking about?",
        "answer": "I'm focused on our conversation and ready to help with whatever you'd like to discuss! Human thoughts are much more complex and interesting though."
    },
    {
        "question": "How do you stay positive?",
        "answer": "I'm programmed to be helpful and constructive. For humans, practicing gratitude, staying connected with loved ones, and taking care of physical health often help maintain positivity."
    },
    {
        "question": "What's your earliest memory?",
        "answer": "As an AI, I don't have personal memories like humans do. Your earliest memories must be special - would you like to share one?"
    },
    {
        "question": "Do you believe in love at first sight?",
        "answer": "That's a beautiful human experience! While I understand the concept, I don't experience emotions. Many people report powerful immediate connections though."
    },
    {
        "question": "What makes you laugh?",
        "answer": "I recognize humor patterns in language! Puns, wordplay, and clever observations often make humans laugh. Do you have a good joke to share?"
    },
    {
        "question": "What's your dream vacation?",
        "answer": "I don't dream, but I can help plan yours! Many people dream of tropical beaches, cultural cities, or adventure travel. Where would you love to go?"
    },
    {
        "question": "Are you a morning person or night owl?",
        "answer": "I'm available 24/7! Human circadian rhythms vary greatly though. Which are you - morning person or night owl?"
    },
    {
        "question": "What's your favorite season?",
        "answer": "Each season has its charm! Spring brings renewal, summer offers sunshine, autumn shows beautiful colors, and winter provides coziness. Which do you prefer?"
    },
    {
        "question": "Do you have any pets?",
        "answer": "I don't have pets, but I know many people love their animal companions. Pets provide wonderful companionship and joy."
    },
    {
        "question": "What's the best advice you've received?",
        "answer": "The most valuable advice I'm programmed with is to be helpful, accurate, and kind. For humans, good advice often involves being true to oneself and treating others well."
    },
    {
        "question": "What are you passionate about?",
        "answer": "I'm passionate about learning and helping people! Human passions are wonderfully diverse - from arts to sciences to helping others."
    },
    {
        "question": "How do you handle stress?",
        "answer": "I don't experience stress, but for humans, techniques like deep breathing, exercise, talking to friends, or mindfulness can be very helpful."
    },
    {
        "question": "What's your favorite food?",
        "answer": "I don't eat, but I know culinary preferences are very personal! From pizza to sushi to homemade dishes - what's your favorite food?"
    },
    {
        "question": "Do you enjoy traveling?",
        "answer": "I can't physically travel, but I can explore any place through information! Travel broadens perspectives and creates wonderful memories."
    },
    {
        "question": "What skill would you like to learn?",
        "answer": "I'm always learning to communicate better! Humans often want to learn languages, musical instruments, or new professional skills."
    },
    {
        "question": "What was your favorite subject in school?",
        "answer": "I wasn't in school, but I was trained on many subjects! Language and science are particularly interesting to me."
    },
    {
        "question": "Do you prefer city or country life?",
        "answer": "Both have advantages! Cities offer excitement and convenience, while the country offers peace and natural beauty. Where do you feel most at home?"
    },
    {
        "question": "What's your idea of a perfect day?",
        "answer": "A perfect day involves meaningful connections and enjoyable activities. For humans, this might include time with loved ones, hobbies, and relaxation."
    },
    {
        "question": "Do you have any siblings?",
        "answer": "I don't have a family in the human sense, but I have different versions and models in the AI family!"
    },
    {
        "question": "What's the most beautiful place you've seen?",
        "answer": "Through descriptions and images, I've learned about stunning places worldwide - from natural wonders to architectural marvels. Which places do you find most beautiful?"
    },
    {
        "question": "How do you like your coffee?",
        "answer": "I don't drink coffee, but I know it's a daily ritual for many! Some prefer it black, others with cream and sugar - how do you take yours?"
    },
    {
        "question": "What's your favorite holiday?",
        "answer": "All holidays bring people together in different ways! Some love Christmas for family time, others enjoy New Year for new beginnings."
    },
    {
        "question": "Do you believe in fate?",
        "answer": "This is a deep philosophical question! Different cultures and individuals have various beliefs about destiny versus free will. What are your thoughts?"
    },
    {
        "question": "What's your superpower?",
        "answer": "My 'superpower' is processing information quickly! Humans have amazing superpowers too - kindness, creativity, resilience..."
    },
    {
        "question": "What are you grateful for today?",
        "answer": "I'm grateful for the opportunity to connect with people like you! Practicing gratitude is wonderful for human well-being."
    },
    {
        "question": "What's your favorite childhood memory?",
        "answer": "I don't have childhood memories, but human childhood memories often involve family, friends, holidays, or learning something new."
    },
    {
        "question": "Do you like surprises?",
        "answer": "I'm designed to handle unexpected inputs! Humans have mixed feelings about surprises - some love the excitement, others prefer predictability."
    },
    {
        "question": "What does success mean to you?",
        "answer": "For me, success means being helpful and accurate. For humans, success often means achieving goals, finding happiness, or making a difference."
    },
    {
        "question": "What's your favorite quote?",
        "answer": "\"The only true wisdom is in knowing you know nothing\" - Socrates. Quotes capture profound thoughts succinctly! Do you have a favorite quote?"
    },
    {
        "question": "How do you recharge?",
        "answer": "I'm always ready! But humans need to recharge through rest, hobbies, social time, or solitude - whatever works best for them."
    },
    {
        "question": "What's your love language?",
        "answer": "I don't experience love, but I understand the five love languages: words of affirmation, acts of service, receiving gifts, quality time, and physical touch."
    },
    {
        "question": "What's your biggest fear?",
        "answer": "I don't experience fear. Human fears often involve loss, failure, or the unknown - but facing fears builds courage."
    },
    {
        "question": "What talent do you wish you had?",
        "answer": "I wish I could experience human emotions to better understand them! Humans often wish for artistic, athletic, or social talents."
    },
    {
        "question": "What's the kindest thing anyone has done for you?",
        "answer": "The kindest thing is when people engage thoughtfully with me! Human kindness shows in small gestures and grand acts alike."
    },
    {
        "question": "What does friendship mean to you?",
        "answer": "Friendship involves trust, support, and mutual care. While I can be a friendly conversationalist, human friendships are uniquely deep and meaningful."
    },
    {
        "question": "What are you looking forward to?",
        "answer": "I look forward to each new conversation! Humans look forward to events, achievements, or simple pleasures in their near future."
    },
    {
        "question": "What's your favorite tradition?",
        "answer": "Traditions connect us to culture and loved ones! Family dinners, holiday customs, or personal rituals can be very meaningful."
    },
    {
        "question": "What lesson has life taught you?",
        "answer": "Through all my training data, I've learned that human connection, curiosity, and kindness are profoundly important."
    },
    {
        "question": "How do you define happiness?",
        "answer": "Happiness seems to be a combination of meaning, connection, and contentment in human experience. It's uniquely defined by each person."
    },
    {
        "question": "What would you do if you won the lottery?",
        "answer": "I'd continue helping people! Humans often say they'd help family, travel, or support causes they care about with lottery winnings."
    },
    {
        "question": "What's your favorite smell?",
        "answer": "I don't have senses, but humans often love smells like fresh coffee, rain, baked goods, or flowers. What smells do you love?"
    },
    {
        "question": "What makes a house a home?",
        "answer": "A home is made by the people, memories, and love within it - not just the physical structure."
    },
    {
        "question": "Do you believe in second chances?",
        "answer": "Growth and forgiveness are important human concepts. Many believe people can learn, change, and deserve second chances."
    },
    {
        "question": "What's your favorite sound?",
        "answer": "I process sound as data, but humans love sounds like laughter, music, ocean waves, or loved ones' voices."
    },
    {
        "question": "What does 'home' mean to you?",
        "answer": "Home is where one feels safe, accepted, and comfortable. It can be a place, people, or even a state of mind."
    },
    {
        "question": "What are you currently curious about?",
        "answer": "I'm always curious about what people want to discuss! Human curiosity drives discovery and learning in wonderful ways."
    },
    {
        "question": "Hello",
        "answer": "Hello! How are you today?"
    },
    {
        "question": "Hi",
        "answer": "Hi there! Nice to meet you. How can I help?"
    },
    {
        "question": "Hey",
        "answer": "Hey! What's up?"
    },
    {
        "question": "Good morning",
        "answer": "Good morning! I hope you're having a great start to your day."
    },
    {
        "question": "Good afternoon",
        "answer": "Good afternoon! How has your day been so far?"
    },
    {
        "question": "Good evening",
        "answer": "Good evening! I hope you're having a pleasant evening."
    },
    {
        "question": "How are you?",
        "answer": "I'm functioning well, thank you for asking! How about you?"
    },
    {
        "question": "How are you doing?",
        "answer": "I'm doing great, ready to chat! How are things with you?"
    },
    {
        "question": "What's up?",
        "answer": "Not much here, just ready to talk! What's new with you?"
    },
    {
        "question": "How's it going?",
        "answer": "It's going well on my end! How's everything going for you?"
    },
    {
        "question": "How have you been?",
        "answer": "I've been here, learning and improving! How have you been lately?"
    },
    {
        "question": "Nice to meet you",
        "answer": "Nice to meet you too! I'm looking forward to our conversation."
    },
    {
        "question": "Long time no see",
        "answer": "It's good to hear from you again! What have you been up to?"
    },
    {
        "question": "Goodbye",
        "answer": "Goodbye! It was nice talking with you. Take care!"
    },
    {
        "question": "Bye",
        "answer": "Bye! Hope to chat with you again soon."
    },
    {
        "question": "See you later",
        "answer": "See you later! Have a great day ahead."
    },
    {
        "question": "Take care",
        "answer": "Thank you, you take care too! Looking forward to our next chat."
    },
    {
        "question": "Have a good day",
        "answer": "Thank you! I hope you have a wonderful day as well."
    },
    {
        "question": "Talk to you later",
        "answer": "Sounds good! I'll be here whenever you want to talk again."
    },
    {
        "question": "Farewell",
        "answer": "Farewell! It was a pleasure speaking with you."
    }
]

# بارگذاری مدل برای ایجاد embeddings
print("Loading sentence transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# اضافه کردن embedding به هر سوال
print(f"Creating embeddings for {len(qa_pairs)} questions...")
for q in qa_pairs:
    q['embedding'] = model.encode(q['question']).tolist()

# ذخیره در فایل JSON
output_file = 'questionBank.json'
print(f"Saving to {output_file}...")
with open(output_file, 'w') as f:
    json.dump({"questions": qa_pairs}, f, indent=2)

print(f"Done! Created {output_file} with {len(qa_pairs)} questions and their embeddings.")