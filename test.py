import cProfile
import pstats
import json
from snake_train import SnakeAgent

def main():
    agent = SnakeAgent()
    agent.intense_training(150)  # Vous pouvez ajuster le nombre d'itérations selon vos besoins

if __name__ == "__main__":
    # Profilage de la fonction main
    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    
    # Extraire les statistiques dans un dictionnaire
    profile_data = {}
    for func_name, func_stats in stats.stats.items():
        file_name, line_num, func_name = func_name
        total_calls, primitive_calls, total_time, cumulative_time, callers = func_stats
        
        profile_data[f"{file_name}:{line_num}({func_name})"] = {
            "total_calls": total_calls,
            "primitive_calls": primitive_calls,
            "total_time": total_time,
            "cumulative_time": cumulative_time
        }
    
    # Sauvegarder les résultats dans un fichier JSON
    with open("profiling_results.json", "w") as f:
        json.dump(profile_data, f, indent=4)
    
    # Afficher uniquement les résultats agrégés
    total_time = sum(stat['total_time'] for stat in profile_data.values())
    total_calls = sum(stat['total_calls'] for stat in profile_data.values())
    
    print(f"Profiling completed. Total time: {total_time:.2f} seconds over {total_calls} function calls.")
    print("Detailed results have been saved to 'profiling_results.json'.")