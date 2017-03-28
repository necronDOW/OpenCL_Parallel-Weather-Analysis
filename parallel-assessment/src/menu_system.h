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

MenuSystem* menu_system;
inline void InitMenus()
{
	menu_system = new MenuSystem();

	menu_system->AddScreen("What would you like to do?");
	menu_system->AddScreenOption(0, "Find Minimum");
	menu_system->AddScreenOption(0, "Find Maximum");
	menu_system->AddScreenOption(0, "Find Mean");
	menu_system->AddScreenOption(0, "Toggle Work Group Size");
	menu_system->AddScreenOption(0, "Exit");

	menu_system->AddScreen("Operate using Global or Local memory?");
	menu_system->AddScreenOption(1, "Global");
	menu_system->AddScreenOption(1, "Local");
}

void MinMaxMenu(PRECISION*& A, PRECISION*& B, size_t base_size, bool dir)
{
	menu_system->ShowScreen(1);

	const char* text = (dir) ? "Maximum: %.1f\n\n" : "Minimum: %.1f\n\n";

	int selection = menu_system->GetScreenOptionSelection();
	switch (selection)
	{
		case 1:
			GlobalMinMax(A, B, base_size, dir);
			printf(text, B[0] / 10.0);
			break;
		case 2:
			LocalMinMax(A, B, base_size, dir);
			printf(text, B[0] / 10.0);
			break;
	}
}

inline void MainMenu(PRECISION*& A, PRECISION*& B, size_t base_size, size_t original_size, bool& finished)
{
	menu_system->ShowScreen(0);

	int selection = menu_system->GetScreenOptionSelection();
	switch (selection)
	{
		case 1: case 2:
			MinMaxMenu(A, B, base_size, selection - 1);
			break;
		case 3:
			Sum(A, B, base_size);
			printf("Mean: %.5f\n\n", mean(B[0] / 10.0, original_size));
			break;
		case 4:
			max_wg_size = !max_wg_size;
			printf("Work Group Size = %s\n\n", (max_wg_size) ? "MAX" : "MIN");
			break;
		default:
			finished = true;
			break;
	}
}

#endif