#include <direct.h>

#include <string>
#include "StringUtils.h"
#ifdef _DEBUG
#undef _DEBUG
#include <Python.h>            // Python high level interface API
#define _DEBUG
#else
#include <Python.h>            // Python high level interface API
#endif
#include <vector>              // std::vector
#include <iostream>            // std::cerr, std::cout
#include <cstdio>              // fopen()


/*! \brief Launches python script which should be presented as \p argv[1] with
 *         arguments argv[2]...argv[argc-1]
 *
 *  \param[in] argc - number of \p C++ input arguments including program name.
 *  \param[in] argv - \p C++ arguments. argv[1] - python program name,
 *                    argv[2]...argv[argc-1] - the python script arguments
 */
int launchPythonScriptWithArguments( int argc, char** argv )
{
    // this C++ program name as wchar_t
	wchar_t* program = Py_DecodeLocale( argv[0], NULL );
	if ( program == NULL )
	{
		std::cerr << "Fatal error: cannot decode argv[0]" << std::endl;
		exit(1);
	}

    if ( argc < 2 )
    {
        std::cerr << "Error: the first argument should be python script name" << std::endl;
        return 0;
    }

    // arguments of the python script as wchar_t string
	std::vector<wchar_t*> pyArgv;
	pyArgv.resize( argc - 1 );

	for ( size_t i = 0; i < pyArgv.size(); ++i )
	{
		pyArgv[i] = Py_DecodeLocale( argv[i + 1], NULL );
        if ( pyArgv[i] == NULL )
        {
            std::cerr << "Fatal error: cannot decode argv[" << i << "]" << std::endl;
            exit( 1 ); // maybe return -1 is better?
        }
	}

    // launching PYTHON program
    Py_SetProgramName( program );
	Py_Initialize();
	PySys_SetArgv( static_cast<int>( pyArgv.size() ), pyArgv.data() );
	FILE* file = fopen( argv[1], "r" );
	PyRun_SimpleFile( file, argv[1] );

	if ( Py_FinalizeEx() < 0 )
	{
		exit(120);     // maybe return -1 is better?
	}

    // deallocating memory
	fclose( file );
	PyMem_RawFree( program );
    for ( auto& i : pyArgv )
        PyMem_RawFree( i );

	return 0;
}


//! Function for launching python script as a command using high python API.
int launchPythonScriptAsCommand( int argc, char** argv )
{
    char cwd[128];
    getcwd( cwd, sizeof( cwd ) );

    wchar_t *program = Py_DecodeLocale( argv[0], NULL );
    if ( program == NULL )
    {
        fprintf( stderr, "Fatal error: cannot decode argv[0]\n" );
        exit( 1 );
    }
    Py_SetProgramName( program );  /* optional but recommended */
    Py_Initialize();

    if ( argc > 1 )
    {
        DebugPrintf( "Running python file: %s\n", argv[1] );
        std::string cmd = "src = open('" + std::string( argv[1] ) + "').read(); exec(src)\n";
        PyRun_SimpleString( cmd.c_str() );
    }
    else
    {
        PyRun_SimpleString( "print('Pass the script to run as command line argument')" );
    }

    if ( Py_FinalizeEx() < 0 )
    {
        exit( 120 );
    }
    PyMem_RawFree( program );
    return 0;
}


int main(int argc, char *argv[])
{
	return launchPythonScriptWithArguments( argc, argv );
}

