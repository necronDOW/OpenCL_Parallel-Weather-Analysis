#ifndef menusystem_h
#define menusystem_h

#include <vector>
#include <string>
#include <functional>
#include <iostream>

#include "funcs.h"

class MenuSystem
{
	class Screen
	{
		private:
			const char* description = "";
			std::vector<std::string> options;

		public:
			Screen(const char* _description)
				: description(_description) { }

			void AddOption(std::string text)
			{
				options.push_back(std::to_string(options.size() + 1) + ". " + text);
			}

			std::string Get()
			{
				std::string output = description;
				for (int i = 0; i < options.size(); i++)
					output += "\n\t" + options[i];

				return output;
			}
	};

	enum ScreenFlag
	{
		MAIN = 0
	};

	private:
		int current_screen = 0;
		std::vector<Screen*> screens;

	public:
		void AddScreen(const char* description)
		{
			if (description != "")
				screens.push_back(new Screen(description));
		}

		void AddScreenOption(int index, const char* text)
		{
			if (index >= 0 && index < screens.size() && text != "")
				screens[index]->AddOption(text);
		}

		int GetScreenOptionSelection()
		{
			std::cout << "\nChoice: ";

			std::string input;
			std::cin >> input;
			int selection = -1;

			try { selection = std::stoi(input); }
			catch(...)
			{
				std::cout << "\nInvalid selection, try again. Choice: ";
				return GetScreenOptionSelection();
			}

			std::cout << "\n";
			return selection;
		}

		void ShowScreen(int index)
		{
			if (index >= 0 && index < screens.size())
			{
				current_screen = index;
				std::cout << screens[current_screen]->Get();
			}
		}
};

/* Below is an implementation of the menu system class shown above which allows for the user to interact easily with the program,
   including a main menu, global/local menu and optimization menu. From the menu system, the user can perform any operation they
   desire and even modify program runtime variables to operate differently on the data set. */

MenuSystem* menu_system;
void InitMenus()
{
	menu_system = new MenuSystem();

	menu_system->AddScreen("What would you like to do?");
	menu_system->AddScreenOption(0, "Find Minimum");
	menu_system->AddScreenOption(0, "Find Maximum");
	menu_system->AddScreenOption(0, "Find Mean");
	menu_system->AddScreenOption(0, "Find Standard Deviation");
	menu_system->AddScreenOption(0, "Find Median");
	menu_system->AddScreenOption(0, "Find Upper Quartile");
	menu_system->AddScreenOption(0, "Find Lower Quartile");
	menu_system->AddScreenOption(0, "Toggle Work Group Size");
	menu_system->AddScreenOption(0, "Choose Optimization Mode");
	menu_system->AddScreenOption(0, "Exit");

	menu_system->AddScreen("Operate using Global or Local memory?");
	menu_system->AddScreenOption(1, "Global");
	menu_system->AddScreenOption(1, "Local");

	menu_system->AddScreen("What would you like to optimize for?");
	menu_system->AddScreenOption(2, "Performance");
	menu_system->AddScreenOption(2, "Precision");
}

template<typename T>
void MinMaxMenu(T*& A, T*& B, size_t& base_size, size_t original_size, fp_type division, bool dir)
{
	menu_system->ShowScreen(1);

	const char* text = (dir) ? "Maximum: %.1f\n\n" : "Minimum: %.1f\n\n";

	int selection = menu_system->GetScreenOptionSelection();
	switch (selection)
	{
		case 1:
			GlobalMinMax(A, B, base_size, original_size, dir);
			printf(text, B[0] / division);
			break;
		case 2:
			LocalMinMax(A, B, base_size, original_size, dir);
			printf(text, B[0] / division);
			break;
	}
}

void OptimizeMenu()
{
	menu_system->ShowScreen(2);

	int selection = menu_system->GetScreenOptionSelection();
	switch (selection)
	{
		case 1: optimize_flag = Performance; break;
		case 2: optimize_flag = Precision; break;
	}
}

template<typename T>
void MainMenu(T*& A, T*& B, size_t& base_size, size_t original_size, bool& finished)
{
	menu_system->ShowScreen(0);
	fp_type division = (typeid(T) == typeid(int)) ? 10.0 : 1.0;

	int selection = menu_system->GetScreenOptionSelection();
	switch (selection)
	{
		case 1: case 2:
			MinMaxMenu(A, B, base_size, original_size, division, selection - 1);
			break;
		case 3:
			Sum(A, B, base_size, original_size);
			printf("Mean: %.5f\n\n", mean(B[0] / division, original_size));
			break;
		case 4:
			Sum(A, B, base_size, original_size);
			Variance(A, B, base_size, original_size, mean(B[0], original_size));
			printf("Standard Deviation: %.3f\n\n", sqrt(B[0] / division));
			break;
		case 5:
			printf("Median: %.3f\n\n", source(SortOptim(A, B, base_size, original_size), original_size, 0.5) / division);
			break;
		case 6:
			printf("Upper Quartile: %.3f\n\n", source(SortOptim(A, B, base_size, original_size), original_size, 0.75) / division);
			break;
		case 7:
			printf("Lower Quartile: %.3f\n\n", source(SortOptim(A, B, base_size, original_size), original_size, 0.25) / division);
			break;
		case 8:
			max_wg_size = !max_wg_size;
			wg_size_changed = true;
			printf("Work Group Size = %s\n\n", (max_wg_size) ? "MAX" : "MIN");
			break;
		case 9:
			OptimizeMenu();
			printf("Optimize Mode = %s\n\n", (optimize_flag == Performance) ? "PERFORMANCE" : "PRECISION");
			break;
		default:
			finished = true;
			break;
	}
}

#endif