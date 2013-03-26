/*
 * xtestrunner.h
 *
 *  Created on: Mar 25, 2013
 *      Author: eaglenature@gmail.com
 */

//******************************************************************
/*  example.cpp

#include "TestRunner.h"

CUDATEST(SampleTest, 0)
{
    printf("Sample test code\n");
}
CUDATEST(NextSampleTest, 1)
{
    printf("Next sample test code\n");
}

class TestList
{
protected:
    void Create(int argc, char** argv, std::vector<unsigned int>& testCaseList) {
        testCaseList.push_back(0);
    }
};

int main(int argc, char** argv)
{
    TestSuite<TestList> suite(argc, argv);
    TestRunner::GetInstance().RunAll(suite);
}

*********************************************************************/

#ifndef XTESTRUNNER_H_
#define XTESTRUNNER_H_

#include <cstdio>
#include <memory>
#include <map>
#include <vector>



template <typename TestList>
class TestSuite : public TestList
{
    using TestList::Create;
public:
    TestSuite(int argc, char** argv)
    : _argc(argc)
    , _argv(argv) { TestList::Create(argc, argv, _testCasesIds); }
private:
    std::vector<unsigned int> _testCasesIds;
    int    _argc;
    char** _argv;
    friend class TestRunner;
};


class TestRunner
{
public:
    class TestCaseBase;
private:
    class FactoryBase {
    public:
        virtual ~FactoryBase() {}
        virtual TestCaseBase* Create(unsigned int) = 0;
    };
public:
    class TestCaseBase {
    public:
        TestCaseBase(unsigned int id) : _id(id) {}
        virtual ~TestCaseBase() {}
        virtual void Body() = 0;
        virtual const char* Name() const = 0;
        unsigned int Id() const { return _id; }
    private:
        unsigned int _id;
    };
    class TestCaseInfo {};
    template <typename T>
    class Factory : public FactoryBase {
    public:
        TestCaseBase* Create(unsigned int a) { return new T(a); }
    };
public:
    void RunAll() {
        std::map<unsigned int, FactoryBase*>::iterator it = _factories.begin();
        while (it != _factories.end()) {
            std::auto_ptr<TestCaseBase> test((*it).second->Create((*it).first));
            printf("=====================================================\n");
            printf("    Run: %s\n", test->Name());
            printf("=====================================================\n");
            test->Body();
            ++it;
        }
        printf("=====================================================\n");
    }
    template <typename T>
    void RunSuite(const TestSuite<T>& testSuite) {
        const std::vector<unsigned int>& d = testSuite._testCasesIds;
        std::vector<unsigned int>::const_iterator it;
        for (it = d.begin(); it != d.end(); ++it) {
            Run(*it);
        }
        printf("=====================================================\n");
    }
    template <typename Factory>
    TestCaseInfo CreateInfo(unsigned int id) {
        Factory* t = new Factory();
        RegisterFactory(id, t);
        return TestCaseInfo();
    }
    ~TestRunner() {
        Cleanup();
    }
    static TestRunner& GetInstance() {
        return instance;
    }
private:
    void Run(unsigned int id) {
        std::map<unsigned int, FactoryBase*>::iterator it;
        it = _factories.find(id);
        if (it != _factories.end()) {
            std::auto_ptr<TestCaseBase> test((*it).second->Create(id));
            printf("=====================================================\n");
            printf("    Run: %s\n", test->Name());
            printf("=====================================================\n");
            test->Body();
        } else {
            printf("    Test issued for execution "
                "but not registered: %d\n", id);
        }
    }
    void RegisterFactory(unsigned int u, FactoryBase* f) {
        _factories.insert(std::make_pair(u, f));
    }
    void Cleanup() {
        std::map<unsigned int, FactoryBase*>::iterator it;
        for (it = _factories.begin();
             it != _factories.end(); ++it) {
            if ((*it).second) delete (*it).second;
        }
        _factories.clear();
    }
    TestRunner() {}
    TestRunner(const TestRunner& runner);
    TestRunner& operator=(const TestRunner& runner);
private:
    static TestRunner instance;
    std::map<unsigned int, FactoryBase*>  _factories;
};

TestRunner TestRunner::instance;

//******************************************************************
#define CUDATEST(TestCaseName, uid)                                \
class TestCaseName##_##uid: public TestRunner::TestCaseBase {      \
private:                                                           \
    TestCaseName##_##uid(unsigned int a);                          \
    TestCaseName##_##uid(const TestCaseName##_##uid&);             \
    TestCaseName##_##uid& operator=(const TestCaseName##_##uid&);  \
    virtual void Body();                                           \
    virtual const char* Name() const;                              \
    static TestRunner::TestCaseInfo info;                          \
    friend class TestRunner;                                       \
};                                                                 \
TestRunner::TestCaseInfo TestCaseName##_##uid::info =              \
    TestRunner::GetInstance().CreateInfo<                          \
        TestRunner::Factory<                                       \
            TestCaseName##_##uid> >((uid));                        \
TestCaseName##_##uid::TestCaseName##_##uid(unsigned int a)         \
    : TestRunner::TestCaseBase(a) {}                               \
const char* TestCaseName##_##uid::Name() const {                   \
    return #TestCaseName;                                          \
}                                                                  \
void TestCaseName##_##uid::Body()
//******************************************************************

#endif /* XTESTRUNNER_H_ */
