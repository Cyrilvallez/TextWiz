import os
import gc
import argparse
import time
import logging

import torch
import numpy as np

from .textwiz import HFModel, loader, warnings_suppressor, utils

# Remove warning when tokenizing sequences longer than expected: we know we are doing it!
logger = logging.getLogger('transformers.tokenization_utils_base')
logger.addFilter(warnings_suppressor.LoggingFilter("Token indices sequence length is longer than the specified maximum sequence length for this model"))


# Random long text about monkeys (thanks ChatGPT!!)
LARGE_TEXT = """Title: Monkeys: Nature's Pranksters, Social Geniuses, and Ecological Wonders

Introduction

Monkeys, the charismatic and diverse members of the primate order, have long held a special place in the annals of our fascination with the animal kingdom. With their playful antics, astonishing intelligence, and complex social structures, they serve as a source of both joy and profound scientific inquiry. In this comprehensive exploration, we embark on a deep dive into the world of monkeys, spanning their evolutionary history, classifications, ecological roles, social dynamics, communication methods, and the pressing need for conservation. These captivating creatures offer insights into the intricacies of the natural world, our own evolutionary heritage, and the urgent importance of preserving biodiversity.

I. Evolutionary Origins

To understand the world of monkeys, we must embark on a journey through their evolutionary past, a tapestry that stretches back millions of years. Monkeys are part of the grand order of Primates, and their lineage is interwoven with the broader history of these remarkable mammals.

A. Primate Origins

The story of primates, including monkeys, begins around 60 million years ago. At that time, the world was a vastly different place, dominated by the reign of dinosaurs. It was during this period of Earth's history that the first primates, known as prosimians, emerged. These small, tree-dwelling mammals exhibited several characteristics that would become hallmarks of all primates: grasping hands and feet, forward-facing eyes for stereoscopic vision, and an enlarged brain relative to body size. These adaptations suited them for life in the trees, where they foraged for insects and fruits.

B. The Emergence of Monkeys

Around 35 million years ago, a significant split occurred within the primate family tree, leading to the emergence of two major groups: New World monkeys (Platyrrhini) and Old World monkeys (Catarrhini). This evolutionary divergence set in motion a cascade of adaptations that would result in the striking diversity of monkeys we see today.

The division between New World and Old World monkeys was not merely a matter of geographical separation but also marked significant differences in physical traits and behaviors. New World monkeys, found in Central and South America, are characterized by their prehensile tails and a variety of adaptations that allow them to thrive in the lush forests of the Americas. Old World monkeys, on the other hand, are residents of Africa, Asia, and parts of Gibraltar, and they have developed their own unique set of features to suit the diverse environments they inhabit.

II. Classification and Diversity

Monkeys are a testament to the incredible diversity that life on Earth can exhibit. With over 260 species distributed across the globe, they vary not only in size and appearance but also in behaviors, diets, and ecological roles. Their classification falls into two major families: Cebidae, housing the New World monkeys, and Cercopithecidae, the home of the Old World monkeys.

A. New World Monkeys (Family: Cebidae)

The New World monkeys, found predominantly in the lush rainforests of Central and South America, present a colorful array of adaptations and lifestyles.

Spider Monkeys: Among the most iconic of New World monkeys, spider monkeys are renowned for their prehensile tails, which serve as a fifth limb for effortless navigation through the treetops. Their remarkably long limbs and prehensile tails are essential tools for finding food and avoiding predators in the dense jungle canopy.

Howler Monkeys: Aptly named for their vocal prowess, howler monkeys are among the loudest animals on Earth. Their distinctive roars resonate through the rainforests, serving both as communication among troop members and territorial declarations. These robust monkeys play a vital role in shaping the dynamics of their forest habitats.

Capuchin Monkeys: Distinguished by their charming white faces, capuchin monkeys are not only visually striking but also exceptionally intelligent. They have been the subjects of numerous scientific studies, demonstrating remarkable problem-solving abilities and tool use. Their keen minds make them adept foragers and adaptors in their challenging forest environments.

Tamarins and Marmosets: Among the smallest of the New World monkeys, tamarins and marmosets exhibit unique social structures and cooperative breeding systems. These diminutive primates challenge our understanding of primate sociality, where extended family members actively participate in the rearing of offspring. Their cooperative lifestyle offers insights into the evolution of social behaviors in primates.

B. Old World Monkeys (Family: Cercopithecidae)

Old World monkeys are a diverse and widespread group, showcasing a range of adaptations and behaviors that allow them to thrive in various environments across Africa, Asia, and parts of Gibraltar.

Baboons: Characterized by their dog-like faces and robust physiques, baboons are iconic Old World monkeys. They have adapted to a wide range of habitats and are known for their complex social structures. Troops of baboons, often numbering several dozen individuals, navigate their environments while juggling intricate hierarchies and social relationships.

Macaques: The macaques are the Old World monkeys that have mastered the art of adaptability. From snowy mountains to lush tropical forests, these monkeys can be found in diverse landscapes across Asia, Africa, and Gibraltar. Their versatility has allowed them to exploit a wide range of resources and ecological niches.

Mandrills: With their vividly colored faces and formidable canines, mandrills claim the title of the largest monkeys in the world. Native to the dense rainforests of Central Africa, these charismatic primates are characterized by their striking appearance and complex social structures.

Langurs and Colobus Monkeys: Langurs and colobus monkeys are leaf-eaters, showcasing an adaptation to a primarily vegetarian diet. Their striking black and white coats, often accompanied by elegant long tails, make them some of the most visually captivating of all primates. They inhabit diverse habitats in Asia and Africa, respectively, and their unique adaptations reflect their distinct evolutionary histories.

III. Ecological Significance

Monkeys, beyond their aesthetic appeal and charismatic behaviors, play an indispensable role in the ecosystems they inhabit. They are often referred to as "keystone species" due to the profound impact their presence or absence can have on the entire ecological web. Let us explore the ways in which monkeys contribute to the balance and health of their habitats.

A. Seed Dispersal

One of the most significant contributions that monkeys make to their ecosystems is seed dispersal. As voracious fruit consumers, they play a vital role in the dispersal of seeds throughout their habitats. Monkeys consume fruits and then transport the seeds, often considerable distances, as they move through the forests. This action is not only crucial for the regeneration of plant populations but also for maintaining the diversity of plant species in their ecosystems. Without the help of monkeys, many plants would struggle to reproduce, potentially leading to shifts in the composition of plant communities.

B. Pruning Vegetation

Monkeys, particularly those that consume leaves and young shoots, act as natural pruners of vegetation. Their selective feeding habits shape the structure of forests by influencing the composition and density of plants. By browsing on specific plant species, monkeys can indirectly influence the presence or absence of other species in their habitats. This pruning behavior has cascading effects on the entire ecosystem, affecting not only plant populations but also the animals that depend on those plants for food and shelter.

C. Predation Control

Some monkey species have a varied diet that includes insects, small mammals, and bird eggs. Their predation on these prey species can help control their populations, preventing overpopulation and mitigating potential harm to local flora and fauna. In this way, monkeys serve as regulators of food webs in their ecosystems, playing a critical role in maintaining ecological balance.

D. Indicator Species

The health and stability of monkey populations can serve as indicators of the overall health of their ecosystems. Declines in monkey populations can be a warning sign of environmental disturbances such as habitat loss, deforestation, climate change, or the spread of diseases. Their sensitivity to changes in their environments makes them valuable sentinels for monitoring the broader ecological conditions of their habitats.

IV. Social Structures and Behavior

One of the most captivating aspects of monkeys is their complex social lives. They live in dynamic societies that exhibit a wide range of behaviors, hierarchies, and adaptations. The structure of monkey social groups can vary significantly between species, reflecting their ecological niches and the challenges they face.

A. Troop Structure

The core of monkey social life revolves around the concept of the "troop." Troops are social groups that consist of multiple individuals, and their size and composition can vary greatly between species. These groups are the foundation of monkey societies, serving as sources of support, protection, and cooperation.

Within these troops, a hierarchy often emerges, dictating access to valuable resources such as food and mates. Dominant individuals occupy higher positions in the hierarchy and enjoy certain privileges. These positions are typically maintained through various displays of dominance, including vocalizations, physical posturing, and, in some cases, even aggression. The dynamics of these hierarchies can be intricate and are influenced by factors such as age, sex, and individual temperament.

B. Mating and Parenting

Monkeys display a fascinating array of mating strategies, ranging from monogamy to polygyny and promiscuity. The particular strategy employed by a species is often influenced by ecological conditions, including resource availability and predation pressure.

In some species, like the monogamous titi monkeys, pairs form strong, long-term bonds and work together to raise their offspring. In contrast, species like the promiscuous bonobos engage in sexual behavior for various social reasons, including conflict resolution and social bonding.

Parenting behaviors among monkeys also vary. In species with cooperative breeding systems, such as marmosets and tamarins, non-breeding individuals within the group actively participate in the care and rearing of the offspring. This cooperative approach to parenting is a testament to the complexity of monkey social structures and highlights their adaptability to different ecological niches.

C. Problem-Solving and Intelligence

Monkeys, particularly those in the Old World monkey group, are renowned for their problem-solving abilities and intelligence. They have been subjects of extensive research in fields such as psychology and primatology.

Capuchin monkeys, for example, have demonstrated remarkable cognitive skills, including the use of tools to extract food from challenging locations. In laboratory settings, they've displayed the ability to understand cause-and-effect relationships and solve intricate puzzles. Such findings not only shed light on the intelligence of monkeys but also provide valuable insights into the evolution of cognitive abilities in primates, including humans.

D. Communication and Language

Communication among monkeys occurs through a diverse array of vocalizations, gestures, and behaviors. These forms of communication serve as a rich tapestry of interaction within their social groups, conveying information about food availability, danger, social bonds, and mating opportunities.

Vocalizations vary widely between species. Howler monkeys, for instance, produce resounding roars that can carry for miles through the forest, serving as both territorial declarations and communication with troop members. In contrast, capuchin monkeys employ a variety of vocalizations and facial expressions to convey their intentions and emotional states. The intricate dance of gestures and postures within monkey societies is a testament to their highly evolved communication systems.

In addition to vocalizations and gestures, some species of monkeys exhibit more complex forms of communication. For instance, vervet monkeys have distinct alarm calls for different predators, enabling troop members to respond appropriately to specific threats. These nuanced forms of communication highlight the depth of understanding and social organization present in monkey societies.

V. Conservation Challenges

Monkeys, despite their adaptability and resilience, face a host of challenges in the modern world. Many of these challenges stem from human activities that disrupt their habitats and threaten their populations. To ensure the long-term survival of these captivating creatures, we must confront and address these conservation challenges.

A. Habitat Loss and Deforestation

One of the most significant threats to monkeys is habitat loss due to deforestation, logging, and urban expansion. The clearing of forests for agriculture and development not only directly reduces available habitat but also fragments existing habitats, isolating populations and limiting genetic diversity. Such fragmentation can make monkeys more vulnerable to diseases and other environmental pressures. The loss of their forest homes also puts them in closer proximity to human communities, increasing the likelihood of conflicts and the potential for retaliatory killings.

B. Illegal Wildlife Trade

Illegal wildlife trade poses a grave danger to monkeys worldwide. They are often targeted for the exotic pet trade and for various traditional medicine markets. The poaching of monkeys not only disrupts natural populations but also inflicts significant suffering on individual animals. Many are captured and transported under horrific conditions, leading to high mortality rates. Those that survive may endure a lifetime of captivity, deprived of their natural behaviors and social structures.

C. Human-Monkey Conflicts

As human populations expand and encroach further into natural habitats, conflicts between humans and monkeys become more frequent. Monkeys may raid crops or urban areas in search of food, leading to human-wildlife conflicts. In response, people may kill or capture monkeys, further threatening their populations. The resolution of such conflicts requires a delicate balance between human needs and the conservation of these vital species.

D. Climate Change

Climate change presents an additional layer of uncertainty for monkey populations. Altered weather patterns, temperature fluctuations, and changing food availability can disrupt the delicate balance of ecosystems, potentially affecting monkey populations and their habitats. As the impacts of climate change continue to unfold, monitoring and adapting to these changes will be crucial for the survival of monkeys and the ecosystems they inhabit.

VI. Conservation Efforts

While the challenges facing monkeys are formidable, there are dedicated conservation efforts aimed at protecting these remarkable creatures and their habitats. These initiatives range from the establishment of protected areas to research and education programs.

A. Protected Areas and Conservation Reserves

The creation of protected areas and conservation reserves is a cornerstone of monkey conservation. These designated areas provide safe havens for monkeys and other wildlife, safeguarding their habitats from the encroachment of human activities. Protected areas also serve as vital research sites, where scientists can study monkey behavior, ecology, and population dynamics. These insights are essential for developing effective conservation strategies.

B. Research and Monitoring

Scientific research plays a crucial role in understanding the needs and behaviors of monkey species. Researchers study everything from their social structures and feeding habits to their response to changing environmental conditions. Monitoring programs track population trends and assess the health of monkey populations, enabling conservationists to identify threats and implement appropriate conservation measures.

C. Community-Based Conservation

Engaging local communities in monkey conservation efforts is often key to success. By involving residents in conservation initiatives, we can build a sense of stewardship and create incentives for protecting monkeys and their habitats. Programs that provide alternative livelihoods and education about the value of wildlife conservation can help reduce human-wildlife conflicts and support the coexistence of people and monkeys.

D. Raising Awareness

Raising awareness about the importance of monkey conservation is essential for garnering support and resources. Public outreach programs, documentaries, and educational initiatives can help people understand the critical role that monkeys play in ecosystems and the urgent need to protect them. Engaging the global community in these efforts fosters a sense of responsibility for the welfare of these remarkable creatures.

VII. Conclusion

Monkeys, these captivating creatures of the wild, offer us a unique lens through which to view the intricacies of the natural world. Their evolutionary journey, complex social structures, communication methods, and ecological significance remind us of the interconnectedness of all life on Earth. As we strive to conserve these extraordinary beings and their habitats, we not only safeguard the diversity of our planet but also enrich our understanding of the delicate tapestry of life that surrounds us.

Monkeys are not merely subjects of fascination; they are ambassadors of the wild, beckoning us to protect and preserve the ecosystems they call home. In doing so, we honor our shared heritage and ensure a brighter future for all living creatures. Our commitment to the conservation of monkeys is not just a moral imperative; it is a testament to our dedication to preserving the wonders of the natural world for generations to come. In their presence, we find inspiration, wonder, and a reminder of the rich tapestry of life that is Earth's greatest treasure.
"""


# For models with max context size of 1024 (e.g. GPT2)
SMALL_CONTEXT_SIZES = [25*i for i in range(1, 41)]
# For other models (we only estimate memory usage up to context size of 2048)
CONTEXT_SIZES = [25*i for i in range(1, 82)]


def memory_usage(past_key_values):
    """Recursively compute the memory footprint of past key values (in bytes).
    """

    if isinstance(past_key_values, torch.Tensor):
        return past_key_values.nelement() * past_key_values.element_size()
    elif isinstance(past_key_values[0], torch.Tensor):
        return sum([x.nelement() * x.element_size() for x in past_key_values])
    else:
        return sum([memory_usage(x) for x in past_key_values])
    

def dtype_category(model, quantization_4bits: bool, quantization_8bits: bool) -> str:
    """Return a string representation of the model dtype."""
    if quantization_4bits:
        return 'int4'
    elif quantization_8bits:
        return 'int8'
    else:
        return str(loader.ALL_MODELS_DTYPES[model]).split('.', 1)[1]



def memory_estimation(model_name: str, quantization_8bits: bool, quantization_4bits: bool, N_repeat: int = 10):
    """Estimate the memory needed to generate text depending on the context size. This function will also check
    if the memory scale with the full context (input size + max_new_tokens), or only with the input size. Indeed,
    in the first forward pass we do not already have the K-V cache, so it needs to be computed. However, in some
    cases the size of the cache is very small compared to the memory needed to compute it the first time, in which
    case the memory only scales with the memory footprint of the first forward pass.

    Parameters
    ----------
    model_name : str
        The name of the model.
    quantization_8bits : bool
        Whether the model will be loaded in 8 bits mode, by default False.
    quantization_4bits : bool
        Whether the model will be loaded in 4 bits mode, by default False.
    N_repeat : int, optional
        How many times to measure the memory footprint, for more precise estimation. We take the mean value of the
        `N_repeat` runs. By default 10
    """

    t0 = time.time()

    # Override quantization for bloom due to its size
    if model_name == 'bloom-176B' and not (quantization_8bits or quantization_4bits):
        quantization_8bits = True

    # Initialize filenames and return if files already exist
    dtype_name = dtype_category(model_name, quantization_4bits=quantization_4bits, quantization_8bits=quantization_8bits)
    filename_memory = os.path.join(utils.DATA_FOLDER, 'memory_estimator', model_name, f'{dtype_name}.json')
    if os.path.exists(filename_memory):
        print(f'It seems like a memory estimation already exists for {model_name} and currently selected dtype.')
        return

    # Load model
    model = HFModel(model_name, quantization_8bits=quantization_8bits, quantization_4bits=quantization_4bits)

    # To avoid possible early stopping on extra eos
    model.extra_eos_tokens = []
        
    gpus = model.get_gpu_devices()
    large_tokens = model.tokenizer.encode(LARGE_TEXT)

    # Initialize dict (this key will be overwritten, but we want it in first for visibility in output file)
    model_memory_consumption = {'only_scale_with_input_size': False}
    
    # select context sizes to use depending on model max context
    input_sizes = SMALL_CONTEXT_SIZES if model.get_context_size() == 1024 else CONTEXT_SIZES

    scale_mode = []

    for input_size in input_sizes:

        prompt = model.tokenizer.decode(large_tokens[:input_size], skip_special_tokens=True)

        results = []
        for k in range(N_repeat):
                
            actual_peaks = {}
            for gpu_rank in gpus:
                torch.cuda.reset_peak_memory_stats(gpu_rank)
                actual_peaks[gpu_rank] = torch.cuda.max_memory_allocated(gpu_rank) / 1024**3

            # Generate 2 new tokens to take into account that we need to estimate the memory of the 
            # computation of past_key_values, and a second forward pass using them (which usually is more costly
            # since the past K-V is large). Subsequent passes will scale linearly with the number of new tokens
            foo = model(prompt, num_return_sequences=1, max_new_tokens=2, min_new_tokens=2, batch_size=1)
            
            memory_used = {}
            for gpu_rank in gpus:
                memory_used[gpu_rank] = (torch.cuda.max_memory_allocated(gpu_rank) / 1024**3) - actual_peaks[gpu_rank]
            
            # Actual largest memory usage peak accross gpus
            max_peak = max(memory_used.values())
            results.append(max_peak)

        # Take the mean value of the N_repeat runs for better estimation
        model_memory_consumption[input_size] = np.mean(results)

        # Estimate the size of the past key values compared to the memory needed to compute them the first time
        # We only need a raw estimate, so we do it only once instead of N_repeat times
        with torch.no_grad():
            prompt_ids = model.tokenizer.encode(prompt, return_tensors='pt').cuda()
            
            for gpu_rank in gpus:
                torch.cuda.reset_peak_memory_stats(gpu_rank)
                actual_peaks[gpu_rank] = torch.cuda.max_memory_allocated(gpu_rank) / 1024**3

            # Single forward pass, caching past key values
            output = model.model(prompt_ids, use_cache=True)

            memory_used = {}
            for gpu_rank in gpus:
                memory_used[gpu_rank] = (torch.cuda.max_memory_allocated(gpu_rank) / 1024**3) - actual_peaks[gpu_rank]
            
        # Actual largest memory usage peak accross gpus
        max_peak = max(memory_used.values())
        # Compute size of past K-V
        past_key_values_memory = memory_usage(output.past_key_values) / 1024**3

        # Our heuristic to estimate if the past_key_values memory size is actually negligible compared to
        # the memory needed for their first computation (an order of magnitude larger means that the first
        # computation of past K-V is actually the memory bottleneck, independently of max_new_tokens)
        if max_peak / past_key_values_memory > 10:
            only_scale_with_input_size = True
        else:
            only_scale_with_input_size = False

        scale_mode.append(only_scale_with_input_size)


    # If this is true for all sizes, then set it to true
    model_memory_consumption['only_scale_with_input_size'] = all(scale_mode)
            
    # Save results
    utils.save_json(model_memory_consumption, filename_memory)

    dt = time.time() - t0

    print(f'Done with {model_name} in {dt/3600:.2f} h!')

    del model
    gc.collect()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Memory estimator')
    parser.add_argument('model', type=str, choices=loader.ALLOWED_MODELS,
                        help='The model to use for memory estimation.')
    parser.add_argument('--int8', action='store_true',
                        help='If given, will estimate the memory footprint of the model quantized to int8.')
    parser.add_argument('--int4', action='store_true',
                        help='If given, will estimate the memory footprint of the model quantized to int4.')
    parser.add_argument('--N', type=int, default=10,
                        help='The number of time to repeat each computation for accurate estimation. By default 10.')
    
    args = parser.parse_args()
    model = args.model
    int8 = args.int8
    int4 = args.int4
    N = args.N

    if int4 and int8:
        raise ValueError('int4 and int8 quantization are mutually exclusive.')
    
    # Do not even attempt to run the script without access to gpus
    if not torch.cuda.is_available():
        raise RuntimeError("I'm begging you, run this benchmark with some GPUs...")

    # Perform the estimation
    memory_estimation(model, int8, int4, N)
