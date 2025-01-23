import numpy as np
from utils.globals import ID_TO_LABEL, TYPES, types

single_singular_lesions = [
    'There is a lesion in the {location} area.',
    'There is a {location} lesion.' ,
    'The lesion is located in the {location} area.',
    'One lesion is located in the {location} region.',
    'The {location} lesion is present.',
    'A lesion is present in the {location} region.',
    'The brain has a {location} tumor.',
    'We found a tumor in the {location} section of the brain.',
    'The scanning shows a brain with a lesion in the {location}.',
    'It can be seen that a {location} tumor is in the MRI.',
    'In the {location} part of the brain a lesion was found.',
    'It is clear that a lesion can be seen in the {location} region.',
    'The patient clearly has a tumor in the {location} part of the brain.',
    'Concise evidence shows that a tumor is present in the {location} area.',
    'The {location} region has a lesion.',
    'The {location} region has a lesion wihtout a doubt.',
    'A {location} tumor is most certainly present.',
    'Magnetic resonance imaging of the patients brain shows a lesion in the {location} region of the brain.',
    'It can be concluded, without a doubt, that the patient has a tumor. Furthermore it is clear the tumor is located in the {location} part.',
    'A tumor has been detected in the {location} lobe of the brain.',
    'There\'s a growth found in the {location} region of the brain.',
    'A tumor has been identified in the {location} area of the brain.',
    'The {location} lobe is afflicted with a tumor.',
    'A tumor has been located in the {location} portion of the brain.',
    'In the {location} region of the brain, a tumor has been discovered.',
    'There\'s a tumor situated in the {location} part of the brain.',
    'The {location} lobe harbors a tumor.',
    'A tumor has been observed in the {location} cortex of the brain.',
    'In the {location} area of the brain, there\'s evidence of a tumor.',
]

single_plural_lesions = [
    'There is {count} lesions in the {location} area.',
    'There is {count} {location} lesions.' ,
    'We found {count} {location} lesions in the mri.',
    'It can be seen that {count} {location} lesions are in the mri.',
    'The scanning shows a brain with {count} lesions in the {location}.',
    '{count} lesions are present in the {location} region.',
    '{count} lesions are located in the {location} area.',
    'It can be seen that {count} {location} tumors are in the MRI.',
    'In the {location} part of the brain {count} lesions was found.',
    'It is clear that {count} lesions can be seen in the {location} region.',
    'The patient clearly has {count} tumors in the {location} part of the brain.',
    'Concise evidence shows that {count} tumors are present in the {location} area.',
    'The {location} region has {count} lesions.',
    'The {location} region has {count} lesions wihtout a doubt.',
    '{count} {location} tumors are most certainly present.',
    'Magnetic resonance imaging of the patients brain shows {count} lesions in the {location} region of the brain.',
    'It can be concluded, without a doubt, that the patient has {count} tumors. Furthermore it is clear the tumors are located in the {location} part.',
    '{count} tumors has been detected in the {location} lobe of the brain.',
    'There\'s {count} growths found in the {location} region of the brain.',
    '{count} tumors has been identified in the {location} area of the brain.',
    'The {location} lobe is afflicted with a tumor.',
    '{count} tumors has been located in the {location} portion of the brain.',
    'In the {location} region of the brain, {count} tumors has been discovered.',
    'There\'s {count} tumors situated in the {location} part of the brain.',
    'The {location} lobe harbors a tumor.',
    '{count} tumors has been observed in the {location} cortex of the brain.',
    'In the {location} area of the brain, there\'s evidence of {count} tumors.',
    ]

duo_singular_lesions = [
    'There are two lesions present, one is located in the {location} and one in the {location2} area.',
    'The scan shows lesions present at the {location} and {location2} regions.',
    'Two lesions are present, one in the {location} and one in the {location2}.',
    'The {location} and {location2} regions both have lesions present.', ## 
    'Lesions are present in the {location} and {location2} area.' ## 
    ]

triplet_singular_lesions = [
    'Multiple lesions are detected in the MRI scan, one in the {location} region, one in the {location2} region and one in the {location3} region.',
    'Three lesions are present in the MRI scan, one in the {location} area, one in the {location2} area and one in the {location3} area. ',
    'The MRI scan shows three lesions, one in the {location} region, one in the {location2} reigion and one in the {location3} region.',
    'The {location}, {location2} and {location3} regions all have lesions present.', ##
    ]

plural_lesion_builder = [
    '{count} lesions are in the {location} area.',
    'there is {count} {location} lesions.' ,
    'we found, among others, {count} {location} lesions in the mri.',
    'it can be seen that there are {count} {location} lesions in the mri.',
    'the scanning shows a brain with {count} lesions in the {location}.',
    '{count} lesions are located in the {location} region.',
    '{count} lesions are present in the {location} region.',
    'it can be seen that, among others, {count} {location} tumors are in the MRI.',
    'in the {location} part of the brain {count} lesions was found.',
    'it is clear that {count} lesions can be seen in the {location} region.',
    'the patient clearly has {count} tumors in the {location} part of the brain.',
    'concise evidence shows that {count} tumors are present in the {location} area.',
    'the {location} region has {count} lesions.',
    'the {location} region has {count} lesions wihtout a doubt.',
    '{count} {location} tumors are most certainly present.',
    'magnetic resonance imaging of the patients brain shows {count} lesions in the {location} region of the brain.',
    'it can be concluded, without a doubt, that the patient has {count} tumors. Furthermore it is clear the tumors are located in the {location} part.'
    '{count} tumors has been detected in the {location} lobe of the brain.',
    'there\'s {count} growths found in the {location} region of the brain.',
    '{count} tumors has been identified in the {location} area of the brain.',
    'the {location} lobe is afflicted with a tumor.',
    '{count} tumors has been located in the {location} portion of the brain.',
    'in the {location} region of the brain, {count} tumors has been discovered.',
    'there\'s {count} tumors situated in the {location} part of the brain.',
    'the {location} lobe harbors a tumor.',
    '{count} tumors has been observed in the {location} cortex of the brain.',
    'in the {location} area of the brain, there\'s evidence of {count} tumors.',
]

singular_lesion_builder = [
    'a lesion is in the {location} area.',
    'there is a {location} lesion.' ,
    'one lesion is located in the {location} region.',
    'the {location} lesion is present.',
    'a lesion is present in the {location} area.',
    'we found a tumor in the {location} section of the brain',
    'the scanning shows a brain with a lesion in the {location}',
    'it can be seen that a {location} tumor is in the MRI',
    'the brain has a {location} tumor.',
    'in the {location} part of the brain a lesion was found.',
    'it is clear that a lesion can be seen in the {location} region.',
    'the patient clearly has a tumor in the {location} part of the brain.',
    'concise evidence shows that a tumor is present in the {location} area.',
    'the {location} region has a lesion.',
    'the {location} region has a lesion wihtout a doubt.',
    'a {location} tumor is most certainly present.',
    'magnetic resonance imaging of the patients brain shows a lesion in the {location} region of the brain.',
    'it can be concluded, without a doubt, that the patient has a tumor. Furthermore it is clear the tumor is located in the {location} part.'
    'a tumor has been detected in the {location} lobe of the brain.',
    'there\'s a growth found in the {location} region of the brain.',
    'a tumor has been identified in the {location} area of the brain.',
    'the {location} lobe is afflicted with a tumor.',
    'a tumor has been located in the {location} portion of the brain.',
    'in the {location} region of the brain, a tumor has been discovered.',
    'there\'s a tumor situated in the {location} part of the brain.',
    'the {location} lobe harbors a tumor.',
    'a tumor has been observed in the {location} cortex of the brain.',
    'in the {location} area of the brain, there\'s evidence of a tumor.',
]

conjunctions = [' Additionally, ',
                ' Furthermore, ',
                ' Moreover, ',
                ' In addition, ',
                ' Also, '
                ] 

gender_strings = [
    'The gender of the patient is {gender}.',
    'Subject is a {gender}.',
    'Scan shows a {gender}.',
    'The gender is {gender}.',
    'The sex of the subject is {gender}.',
    'It is clearly a {gender}.',
    'A human of the {gender} gender can be observed.',
    'This is a {gender} brain.',
    'The person is a {gender}.',
    '{gender} gender has been indicated for the patient.',
    'The patient\'s gender designation is {gender}.',
    '{gender} gender identity is associated with the patient.',
    'The patient\'s gender expression aligns with {gender}.',
    'The patient\'s self-identified gender is {gender}.',
    '{gender} gender category applies to the patient.',
    'The patient\'s gender presentation is consistent with {gender}.',
    'The patient\'s gender marker is {gender}.',
    '{gender} gender classification pertains to the patient.',
    'The patient\'s gender preference is {gender}.',
    '{gender} gender affiliation is attributed to the patient.',
    'The patient\'s gender attribution is {gender}.',
    'The patient\'s gender as identified is {gender}.',
    '{gender} gender characterization applies to the patient.',
    'The patient\'s gender categorization is {gender}.',
    'The patient\'s gender role is that of {gender}.',
    '{gender} gender assignment is acknowledged for the patient.',
    'The patient\'s gender designation corresponds to {gender}.',
    'The patient\'s gender association is with {gender}.',
]

age_string = [
    'The age of the subejct is {age}.',
    'The scan shows a {age} year old brain.',
    'The patient is {age} years old.',
    'This is clearly a person of {age} years.',
    'The person is of the age {age}.',
    'The patient\'s age is {age}.',
    'The individual is in their {age}th year of life.',
    'The patient has reached the age of {age}.',
    'The individual is {age} years young.',
    'The person is {age} years of age.',
    'The age of the patient is {age} years.',
    'The individual is {age} years in age.',
    'The patient is at the age of {age}.',
    'The person has lived {age} years.',
    'The individual has attained the age of {age}.',
    'The patient\'s chronological age is {age}.',
    'The person has completed {age} years of life.',
    'The patient is {age} years of age.',
    'The individual\'s age stands at {age}.',
    'The patient is {age} years into their journey of life.',
    'The person has existed for {age} years.',
    'The patient has accumulated {age} years of experience.',
    'The individual has celebrated {age} birthdays.',
    'The patient\'s age registers at {age} years.',
]

count_to_string = {1 : "one", 2: "two", 3:"three", 4:"four", 5 : "five", 6: "six", 7 : "seven", 8 : "eight", 9 : "nine", 10 : "ten"}
    

def generate_random_gender(data_dict):
    gender_count = len(gender_strings)
    sentence_idx = np.random.randint(0, gender_count)
    return gender_strings[sentence_idx].format(gender=data_dict['gender'])    

def generate_random_age(data_dict):
    age_count = len(age_string)
    sentence_idx = np.random.randint(0, age_count)
    return age_string[sentence_idx].format(age=data_dict['age'])    

def generate_description(data_dict, uniques):
    intro = "The following is true for the current magnetic resonance imaging (MRI) scan."
    age = generate_random_age(data_dict)
    gender = generate_random_gender(data_dict)
    diagnosis = generate_random_locations(uniques)

    return intro + ' ' + age + ' ' + gender + ' ' + diagnosis


def generate_age(age):
  return f"The age of the patient is {age}."

def generate_gender(gender):
  return f"The gender of the subject is " +  gender.lower()

def generate_tumor(tumor):
   return f"\nFindings:\nThe brain MRI demonstrates a lesion in the {tumor}."
   
def generate_location(location):
  return f"There is a lesion in the {location} section."

def generate_location_direction(location_direction):
  return f"\nFindings:\n {location_direction}." 

def generate_texts(data_dict, type):
    if type == TYPES.TUMOR:
        return [generate_tumor(tumor) for tumor in types[type]['labels']]
    elif type == TYPES.GENDER:
        return [f"The patient is a {gender}" for gender in types[type]['labels']]
    elif type == TYPES.LOCATION:
        return [generate_location(loc) for loc in types[type]['labels']]
    elif type == TYPES.LOCATION_DIRECTION:
        return [generate_location_direction(loc_dir) for loc_dir in types[type]['labels']]



def generate_singular_locations(list_of_uniques):
    np.random.shuffle(list_of_uniques)
    locations_count = len(list_of_uniques)
    if locations_count == 1:
        sentence_idx = np.random.randint(0, len(single_singular_lesions))
        return single_singular_lesions[sentence_idx].format(location=ID_TO_LABEL[list_of_uniques[0]])
    if locations_count == 2:
            sentence_idx = np.random.randint(0, len(duo_singular_lesions))
            return duo_singular_lesions[sentence_idx].format(
                location=ID_TO_LABEL[list_of_uniques[0]], 
                location2=ID_TO_LABEL[list_of_uniques[1]])
    if locations_count == 3:
        sentence_idx = np.random.randint(0, len(triplet_singular_lesions))
        return triplet_singular_lesions[sentence_idx].format(
            location=ID_TO_LABEL[list_of_uniques[0]], 
            location2=ID_TO_LABEL[list_of_uniques[1]], 
            location3=ID_TO_LABEL[list_of_uniques[2]])
    else:
        sentence = ''
        for i in range(len(list_of_uniques)):
            if i == 0:
                sentence_idx = np.random.randint(0, len(single_singular_lesions))
                sentence +=  single_singular_lesions[sentence_idx].format(location=ID_TO_LABEL[list_of_uniques[i]])
            
            else:
                sentence_idx = np.random.randint(0, len(singular_lesion_builder))
                sentence +=  singular_lesion_builder[sentence_idx].format(location=ID_TO_LABEL[list_of_uniques[i]])
                
            if i != len(list_of_uniques) - 1:
                conjunction_idx = np.random.randint(0, len(conjunctions))
                sentence += conjunctions[conjunction_idx]
        return sentence
    
def generate_plural_locations(uniques, list_of_uniques):
    sentence = ''
    for i in range(len(list_of_uniques)):
        if i == 0:
            location_id = list_of_uniques[i]
            sentence_array = single_singular_lesions if uniques[location_id] == 1 else single_plural_lesions
            sentence_idx = np.random.randint(0, len(sentence_array))
            count = count_to_string[uniques[location_id]]
            sentence +=  sentence_array[sentence_idx].format(count=count, location=ID_TO_LABEL[location_id])
        else:
            location_id = list_of_uniques[i]
            sentence_array = singular_lesion_builder if uniques[location_id] == 1 else plural_lesion_builder
            sentence_idx = np.random.randint(0, len(sentence_array))
            count = count_to_string[uniques[location_id]]
            sentence +=  sentence_array[sentence_idx].format(count=count, location=ID_TO_LABEL[location_id])
            
        if i != len(list_of_uniques) - 1:
            conjunction_idx = np.random.randint(0, len(conjunctions))
            sentence += conjunctions[conjunction_idx]
    return sentence
    
# uniques is a dict where the key is the id of the location type and the value is the count
def generate_random_locations(uniques):
    if isinstance(uniques, list) or isinstance(uniques, np.ndarray):
        np.random.shuffle(uniques)
        return generate_singular_locations(uniques)
    
    # if no lesion types appear more than once
    list_of_uniques = list(uniques.keys())
    np.random.shuffle(list_of_uniques)

    if sum(uniques.values()) == len(uniques.keys()):
        return generate_singular_locations(list_of_uniques)
    
    else:
        return generate_plural_locations(uniques, list_of_uniques)
        
def generate_enum_from_string(type):
   if type == 'location':
      return TYPES.LOCATION
   if type == 'gender':
      return TYPES.GENDER
   if type == 'tumor':
      return TYPES.TUMOR
   if type == 'location_direction':
      return TYPES.LOCATION_DIRECTION
   if type == 'age':
       return TYPES.AGE