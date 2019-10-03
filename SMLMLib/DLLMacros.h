#pragma once

#define DLL_CALLCONV __cdecl
#ifdef SMLM_EXPORTS
	#define DLL_EXPORT __declspec(dllexport) 
#else
	#define DLL_EXPORT __declspec(dllimport)
#endif

// Support C for matlab imports
#ifdef __cplusplus
#define CDLL_EXPORT extern "C" DLL_EXPORT
#define CONST_REF(_Type,_Name) const _Type& _Name
#define STRUCT 
#else
#define STRUCT struct
#define CDLL_EXPORT DLL_EXPORT
#define CONST_REF(_Type, _Name) const _Type* _Name
#endif
