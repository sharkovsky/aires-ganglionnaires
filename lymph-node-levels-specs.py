level_specs = {
    'level_ia_left': {
        'inferior border': [{'border': ['inferior border'], 'structure': 'platysma_left'}], #Pixel le plus haut du Plastysma
        'superior border': [{'border': ['superior border'], 'structure': 'submandibular_gland_left'}], #Pixel le plus bas
        'posterior border': [{'border': ['posterior border'], 'structure': 'hyoid'}], #pixel le plus postérieure
        'anterior border': [{'border': ['anterior border'], 'structure': 'skull'}], #pixel le plus antérieure
        'left border': [{'border': ['left border'], 'structure': 'digastric_left'}]
        'right border': [{'border': ['right border'], 'structure': 'digastric_right'}] },

    'level_ia_right': {
        'inferior border': [{'border': ['inferior border'], 'structure': 'platysma_right'}], #Pixel le plus haut du Plastysma
        'superior border': [{'border': ['superior border'], 'structure': 'submandibular_gland_right'}], #Pixel le plus bas
        'posterior border': [{'border': ['posterior border'], 'structure': 'hyoid'}], #pixel le plus postérieure
        'anterior border': [{'border': ['anterior border'], 'structure': 'skull'}], #pixel le plus antérieure
#        'lateral border': [{'border': ['lateral border'], 'structure': 'digastric_right'}]
        'left border': [{'border': ['left border'], 'structure': 'digastric_left'}]
        'right border': [{'border': ['right border'], 'structure': 'digastric_right'}] },

    'level_ib_left': {
        'inferior border': [{'border': ['inferior border'], 'structure': 'hyoid'}], #pixel le plus bas
        'superior border': [{'border': ['superior border'], 'structure': 'submandibular_gland_left'}], #Pixel le plus haut
        'posterior border': [{'border': ['posterior border'], 'structure': 'submandibular_gland_left'}], #pixel le plus postérieure
        'anterior border': [{'border': ['anterior border'], 'structure': 'skull'}], #pixel le plus antérieure
        'left border': [{'border': ['left border'], 'structure': 'plastysma_left'}]
        'right border': [{'border': ['right border'], 'structure': 'digastric_left'}] #pixel les plus ) gauche
},

    'level_ib_right': {
        'inferior border': [{'border': ['inferior border'], 'structure': 'hyoid'}], #pixel le plus bas
        'superior border': [{'border': ['superior border'], 'structure': 'submandibular_gland_right'}], #Pixel le plus bas
        'posterior border': [{'border': ['posterior border'], 'structure': 'submandibular_gland_left'}], #pixel le plus postérieure
        'anterior border': [{'border': ['anterior border'], 'structure': 'structure': 'skull'}], #pixel le plus antérieure
#        'lateral border': [{'border': ['lateral border'], 'structure': 'medial_surface_of_mandible'}]
        'left border': [{'border': ['left border'], 'structure': 'digastric_right'}] #pixels les plus à droites
        'right border': [{'border': ['right border'], 'structure': 'plastysma_right'}] },

    'level_ii_left': {
        'inferior border': [{'border': ['inferior border'], 'structure': 'hyoid'}], #pixel le plus bas
        'superior border': [{'border': ['superior border'], 'structure': 'vertebrae_C1'}], #pixel le plus bas
        'posterior border': [{'border': ['posterior border'], 'structure': 'sternocleidomastoid_left'}], #pixel le plus postérieur
        'anterior border': [{'border': ['anterior border'], 'structure': 'submandibular_gland_left'}], #pixel le plus postérieure
        'left border': [{'border': ['left border'], 'structure': 'sternocleidomastoid_left'}], #pixel le plus à droite (ne pas prendre le muscle avec)
        'right border': [{'border': ['right border'], 'structure': 'internal_carotid_artery_left'}] #coupe axiale du pixel le plus à droite de la carotide
        },

    'level_ii_right': {
        'inferior border': [{'border': ['inferior border'], 'structure': 'hyoid'}], #pixel le plus bas
        'superior border': [{'border': ['superior border'], 'structure': 'vertebrae_C1'}], #pixel le plus bas
        'posterior border': [{'border': ['posterior border'], 'structure': 'sternocleidomastoid_right'}], #pixel le plus postérieur
        'anterior border': [{'border': ['anterior border'], 'structure': 'submandibular_gland_left'}], #pixel le plus postérieure
        'left border': [{'border': ['left border'], 'structure': 'internal_carotid_artery_right'}] #coupe axiale du pixel le plus à gauche de la carotide
        'right border': [{'border': ['right border'], 'structure': 'structure': 'sternocleidomastoid_right'}], #pixel le plus à gauche (ne pas prendre le muscle avec)
        },

    'level_iii_left': {
        'inferior border': [{'border': ['inferior border'], 'structure': 'cricoid_cartilage'}], #pixel le plus bas
        'superior border': [{'border': ['superior border'], 'structure': 'hyoid'}], #pixel le plus bas
        'posterior border': [{'border': ['posterior border'], 'structure': 'sternocleidomastoid_left'}], #pixel le plus postérierue
        'anterior border': [{'border': ['anterior border'], 'structure': 'sternocleidomastoid_left'}], #pixel le plus antérieure
        'left border': [{'border': ['left border'], 'structure': 'sternocleidomastoid_left'}], #pixel touchant le SCM
        'right border': [{'border': ['right border'], 'structure': 'common_carotid_artery_left'}] #coupe axiale de son pixel le plus à droite 
        },

    'level_iii_right': {
        'inferior border': [{'border': ['inferior border'], 'structure': 'cricoid_cartilage'}], #pixel le plus bas
        'superior border': [{'border': ['superior border'], 'structure': 'hyoid'}], #pixel le plus bas
        'posterior border': [{'border': ['posterior border'], 'structure': 'sternocleidomastoid_right'}], #pixel le plus postérierue
        'anterior border': [{'border': ['anterior border'], 'structure': 'sternocleidomastoid_right'}], #pixel le plus antérieure
        'left border': [{'border': ['left border'], 'structure': 'sternocleidomastoid_right'}], #pixel touchant le SCM
        'right border': [{'border': ['right border'], 'structure': 'common_carotid_artery_right'}] #coupe axiale de son pixel le plus à gauche
        },
    
'level_iv_left': {
        'inferior border': [{'border': ['inferior border'], 'structure': 'sternum'}], #pixel le plus haut
        'superior border': [{'border': ['superior border'], 'structure': 'cricoid_cartilage'}], #pixel le plus bas
        'posterior border': [{'border': ['posterior border'], 'structure': 'anterior_scalene_left'}],
        'anterior border': [{'border': ['anterior border'], 'structure': 'sternocleidomastoid_left'}],
        'left border': [{'border': ['left border'], 'structure': ''}],
        'right border': [{'border': ['right border'], 'structure': 'common_carotid_artery_left'}] #coupe axiale de son pixel le plus à droite 
        },

    'level_iv_right': {
        'inferior border': [{'border': ['inferior border'], 'structure': 'sternum'}], #pixel le plus haut
        'superior border': [{'border': ['superior border'], 'structure': 'cricoid_cartilage'}], #pixel le plus bas
        'posterior border': [{'border': ['posterior border'], 'structure': 'anterior_scalene_right'}],
        'anterior border': [{'border': ['anterior border'], 'structure': 'sternocleidomastoid_right'}],
        'left border': [{'border': ['left border'], 'structure': 'common_carotid_artery_right'}] #coupe axiale de son pixel le plus à gauche
        'right border': [{'border': ['right border'], 'structure': ''}]
        },

    'level_v_left': {
        'inferior border': [{'border': ['inferior border'], 'structure': 'clavicula_left'}], 'superior border': [{'border': ['superior border'], 'structure': 'hyoid'}],
        'posterior border': [{'border': ['posterior border'], 'structure': 'trapezius_left'}], 'anterior border': [{'border': ['anterior border'], 'structure': 'sternocleidomastoid_left'}],
        'left border': [{'border': ['left border'], 'structure': 'platysma_left'}],
        'right border': [{'border': ['right border'], 'structure': 'anterior_scalene_left'}] },

    'level_v_right': {
        'inferior border': [{'border': ['inferior border'], 'structure': 'clavicula_right'}], 'superior border': [{'border': ['superior border'], 'structure': 'hyoid'}],
        'posterior border': [{'border': ['posterior border'], 'structure': 'trapezius_right'}], 'anterior border': [{'border': ['anterior border'], 'structure': 'sternocleidomastoid_right'}],
        'left border': [{'border': ['left border'], 'structure': 'anterior_scalene_right'}], 'right border': [{'border': ['right border'], 'structure': 'platysma_right'}]
        },

    'level_via_left': {
        'inferior border': [{'border': ['inferior border'], 'structure': 'sternum'}], #pixel le plus haut
        'superior border': [{'border': ['superior border'], 'structure': 'hyoid'}], #pixel le plus bas
        'posterior border': [{'border': ['posterior border'], 'structure': 'thyrohyoid_left'}], #pixel le plus antérieure
        'anterior border': [{'border': ['anterior border'], 'structure': 'platysma_left'}], 'left border': [{'border': ['left border'], 'structure': 'sternocleidomastoid_left'}], 'right border': [{'border': ['right border'], 'structure': 'sternocleidomastoid_right'}]
        },

    'level_via_right': {
        'inferior border': [{'border': ['inferior border'], 'structure': 'sternum'}], #pixel le plus haut
        'superior border': [{'border': ['superior border'], 'structure': 'hyoid'}], #pixel le plus bas
        'posterior border': [{'border': ['posterior border'], 'structure': 'thyrohyoid_right'}], #pixel le plus antérieure
        'anterior border': [{'border': ['anterior border'], 'structure': 'platysma_right'}], 'left border': [{'border': ['left border'], 'structure': 'sternocleidomastoid_left'}], 'right border': [{'border': ['right border'], 'structure': 'sternocleidomastoid_right'}]
        },

    'level_vib_left': {
        'inferior border': [{'border': ['inferior border'], 'structure': 'sternum'}], #pixel le plus haut
        'superior border': [{'border': ['superior border'], 'structure': 'thyroid_cartilage'}], #pixel le plus bas
        'posterior border': [{'border': ['posterior border'], 'structure': 'prevertebral_left'}], #pixel les plu santérieures
        'anterior border': [{'border': ['anterior border'], 'structure': 'thyrohyoid_left'}], #pixel le plus postérieure
        'left border': [{'border': ['left border'], 'structure': 'common_carotid_artery_left'}], #pixel le plus à droite
        'right border': [{'border': ['right border'], 'structure': 'trachea'}]
        },

    'level_vib_right': {
        'inferior border': [{'border': ['inferior border'], 'structure': 'sternum'}], #pixel le plus haut
        'superior border': [{'border': ['superior border'], 'structure': 'thyroid_cartilage'}], #pixel le plus bas
        'posterior border': [{'border': ['posterior border'], 'structure': 'prevertebral_right'}], #pixel les plu santérieures
        'anterior border': [{'border': ['anterior border'], 'structure': 'thyrohyoid_right'}], #pixel le plus postérieure
        'left border': [{'border': ['left border'], 'structure': 'trachea'}]
        'right border': [{'border': ['right border'], 'structure': ' common_carotid_artery_right'}], #pixel le plus à gauche 
        }
    }

